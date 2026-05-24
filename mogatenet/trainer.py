import os
import time

import torch
from monai.metrics.utils import do_metric_reduction
from monai.utils.enums import MetricReduction
from tensorboardX import SummaryWriter
from tqdm import tqdm

from utils.utils import AverageMeter


_SKIP_BATCH_KEYS = {
    "fold",
    "image_meta_dict",
    "label_meta_dict",
    "foreground_start_coord",
    "foreground_end_coord",
    "image_transforms",
    "label_transforms",
}


def _get_device(model):
    return next(model.parameters()).device


def _move_batch_to_device(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
        if key not in _SKIP_BATCH_KEYS
    }


def _reduce_loss(loss):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        loss = loss.detach().clone()
        torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
        loss = loss / torch.distributed.get_world_size()
    return loss


def train_epoch(model, loader, optimizer, epoch, args, loss_func, writer=None):
    model.train()

    device = _get_device(model)
    start_time = time.time()
    run_loss = AverageMeter()

    for idx, batch in enumerate(loader):
        batch = _move_batch_to_device(batch, device)
        image = batch["image"]
        target = batch["label"]

        optimizer.zero_grad(set_to_none=True)

        outputs = model(image)
        loss = loss_func(outputs, target)

        loss.backward()
        optimizer.step()

        reduced_loss = _reduce_loss(loss)
        run_loss.update(reduced_loss.item(), n=args.batch_size)

        if args.rank == 0:
            print(
                f"Epoch {epoch}/{args.max_epochs - 1} "
                f"[{idx + 1}/{len(loader)}] "
                f"loss: {run_loss.avg:.4f} "
                f"time: {time.time() - start_time:.2f}s"
            )

        start_time = time.time()

    optimizer.zero_grad(set_to_none=True)
    return run_loss.avg


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    save_dict = {
        "epoch": epoch,
        "best_acc": best_acc,
        "state_dict": state_dict,
    }

    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()

    save_path = os.path.join(args.logdir, filename)
    torch.save(save_dict, save_path)
    print(f"Saving checkpoint: {save_path}")


class Trainer:
    def __init__(self, args, train_loader, loss_func, validator=None):
        self.args = args
        self.train_loader = train_loader
        self.loss_func = loss_func
        self.validator = validator

    def train(self, model, optimizer, scheduler=None, start_epoch=0):
        args = self.args
        writer = None

        if args.logdir is not None and args.rank == 0:
            writer = SummaryWriter(log_dir=args.logdir)
            print(f"Writing TensorBoard logs to {args.logdir}")

        best_mean_dice = 0.0
        best_metric = None

        for epoch in range(start_epoch, args.max_epochs):
            if args.distributed:
                self.train_loader.sampler.set_epoch(epoch)
                torch.distributed.barrier()

            if args.rank == 0:
                print(f"Rank {args.rank} | {time.ctime()} | Epoch {epoch}")

            epoch_time = time.time()

            train_loss = train_epoch(
                model=model,
                loader=self.train_loader,
                optimizer=optimizer,
                epoch=epoch,
                args=args,
                loss_func=self.loss_func,
                writer=writer,
            )

            if args.rank == 0:
                print(
                    f"Final training {epoch}/{args.max_epochs - 1} "
                    f"loss: {train_loss:.4f} "
                    f"time: {time.time() - epoch_time:.2f}s"
                )

                if writer is not None:
                    writer.add_scalar("train_loss", train_loss, epoch)

            if (epoch + 1) % args.val_every == 0 and self.validator is not None:
                if args.distributed:
                    torch.distributed.barrier()

                val_time = time.time()
                val_metric = self.validator.run()
                mean_dice = self.validator.metric_dice_avg(val_metric)

                if args.rank == 0:
                    print(
                        f"Final validation {epoch}/{args.max_epochs - 1} "
                        f"metric: {val_metric} "
                        f"mean_dice: {mean_dice:.6f} "
                        f"time: {time.time() - val_time:.2f}s"
                    )

                    if writer is not None:
                        for name, value in val_metric.items():
                            writer.add_scalar(name, value, epoch)
                        writer.add_scalar("mean_dice", mean_dice, epoch)

                    is_new_best = mean_dice > best_mean_dice
                    if is_new_best:
                        print(f"New best mean Dice: {best_mean_dice:.6f} -> {mean_dice:.6f}")
                        best_mean_dice = mean_dice
                        best_metric = val_metric

                        save_checkpoint(
                            model=model,
                            epoch=epoch,
                            args=args,
                            filename="model.pt",
                            best_acc=best_mean_dice,
                            optimizer=optimizer,
                            scheduler=scheduler,
                        )

                    with open(os.path.join(args.logdir, "log.txt"), "a", encoding="utf-8") as f:
                        f.write(f"epoch: {epoch + 1}, metric: {val_metric}\n")
                        f.write(f"epoch: {epoch + 1}, mean_dice: {mean_dice}\n")
                        f.write(f"epoch: {epoch + 1}, best_metric: {best_metric}\n")
                        f.write(f"epoch: {epoch + 1}, best_mean_dice: {best_mean_dice}\n")
                        f.write("*" * 20 + "\n")

            if scheduler is not None:
                scheduler.step()

        if writer is not None:
            writer.close()

        if args.rank == 0:
            print(f"Training finished. Best metric: {best_metric}")

        return best_metric


class Validator:
    def __init__(
        self,
        args,
        model,
        val_loader,
        class_list,
        metric_functions,
        sliding_window_infer=None,
        post_label=None,
        post_pred=None,
    ):
        self.args = args
        self.model = model
        self.val_loader = val_loader
        self.class_list = class_list
        self.metric_functions = metric_functions
        self.sliding_window_infer = sliding_window_infer
        self.post_label = post_label
        self.post_pred = post_pred

    def metric_dice_avg(self, metric):
        dice_values = [value for name, value in metric.items() if "dice" in name.lower()]
        if len(dice_values) == 0:
            return 0.0
        return sum(dice_values) / len(dice_values)

    def run(self):
        self.model.eval()
        device = _get_device(self.model)

        metric_sums = None
        not_nan_sums = None

        class_metric_names = []
        for metric_name, _ in self.metric_functions:
            for class_name in self.class_list:
                class_metric_names.append(f"{class_name}_{metric_name}")

        for _, batch in tqdm(enumerate(self.val_loader), total=len(self.val_loader)):
            batch = _move_batch_to_device(batch, device)
            image = batch["image"]
            label = batch["label"]

            with torch.no_grad():
                if self.sliding_window_infer is not None:
                    logits = self.sliding_window_infer(image, self.model)
                else:
                    logits = self.model(image)

                if self.post_label is not None:
                    label = self.post_label(label)

                if self.post_pred is not None:
                    logits = self.post_pred(logits)

                batch_metric_sums = []
                batch_not_nan_sums = []

                for _, metric_fn in self.metric_functions:
                    metric = metric_fn(y_pred=logits, y=label)
                    metric, not_nan = do_metric_reduction(metric, MetricReduction.MEAN_BATCH)

                    metric = metric.to(device)
                    not_nan = not_nan.to(device)

                    batch_metric_sums.append(metric * not_nan)
                    batch_not_nan_sums.append(not_nan)

                    if hasattr(metric_fn, "reset"):
                        metric_fn.reset()

                batch_metric_sums = torch.cat([m.flatten() for m in batch_metric_sums])
                batch_not_nan_sums = torch.cat([n.flatten() for n in batch_not_nan_sums])

                if metric_sums is None:
                    metric_sums = batch_metric_sums
                    not_nan_sums = batch_not_nan_sums
                else:
                    metric_sums += batch_metric_sums
                    not_nan_sums += batch_not_nan_sums

        if metric_sums is None:
            return {name: 0.0 for name in class_metric_names}

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
            torch.distributed.all_reduce(metric_sums, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(not_nan_sums, op=torch.distributed.ReduceOp.SUM)

        not_nan_sums = torch.where(not_nan_sums == 0, torch.ones_like(not_nan_sums), not_nan_sums)
        final_metrics = metric_sums / not_nan_sums

        return {
            name: value
            for name, value in zip(class_metric_names, final_metrics.detach().cpu().tolist())
        }
