import os
import json
import math
import numpy as np
import torch
from monai import transforms, data


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Distributed package is not available.")
            num_replicas = torch.distributed.get_world_size()

        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Distributed package is not available.")
            rank = torch.distributed.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.make_even = make_even
        self.epoch = 0

        self.num_samples = int(math.ceil(len(self.dataset) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank:self.total_size:self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: self.total_size - len(indices)]
                else:
                    extra_ids = np.random.randint(
                        low=0,
                        high=len(indices),
                        size=self.total_size - len(indices),
                    )
                    indices += [indices[i] for i in extra_ids]

            assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def _join_path(base_dir, path):
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


def datafold_read(datalist, basedir, fold=0, key="training"):
    with open(datalist, "r") as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for item in json_data:
        if "image" in item:
            if isinstance(item["image"], list):
                item["image"] = [_join_path(basedir, x) for x in item["image"]]
            elif isinstance(item["image"], str):
                item["image"] = _join_path(basedir, item["image"])

        if "label" in item and isinstance(item["label"], str):
            item["label"] = _join_path(basedir, item["label"])

    train_files = []
    val_files = []

    for item in json_data:
        if item.get("fold", None) == fold:
            val_files.append(item)
        else:
            train_files.append(item)

    return train_files, val_files


def get_train_transform(args):
    return transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]),
            transforms.CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                k_divisible=[args.roi_x, args.roi_y, args.roi_z],
            ),
            transforms.RandSpatialCropd(
                keys=["image", "label"],
                roi_size=[args.roi_x, args.roi_y, args.roi_z],
                random_size=False,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            transforms.NormalizeIntensityd(
                keys="image",
                nonzero=True,
                channel_wise=True,
            ),
            transforms.RandScaleIntensityd(
                keys="image",
                factors=0.1,
                prob=1.0,
            ),
            transforms.RandShiftIntensityd(
                keys="image",
                offsets=0.1,
                prob=1.0,
            ),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )


def get_val_transform(args):
    return transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]),
            transforms.NormalizeIntensityd(
                keys="image",
                nonzero=True,
                channel_wise=True,
            ),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )


def get_loader(args):
    data_dir = getattr(args, "data_dirs", "./data/TrainingData")
    datalist_json = getattr(args, "json_list", "./data/brats2020_datajson.json")

    train_files, val_files = datafold_read(
        datalist=datalist_json,
        basedir=data_dir,
        fold=args.fold,
    )

    train_ds = data.Dataset(
        data=train_files,
        transform=get_train_transform(args),
    )

    distributed = getattr(args, "distributed", False)
    train_sampler = Sampler(train_ds) if distributed else None

    train_loader = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        sampler=train_sampler,
        pin_memory=False,
    )

    val_ds = data.Dataset(
        data=val_files,
        transform=get_val_transform(args),
    )

    val_sampler = Sampler(val_ds, shuffle=False) if distributed else None

    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        sampler=val_sampler,
        pin_memory=False,
    )

    return train_loader, val_loader
