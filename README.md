# MoGateNet

**"MoGateNet: Modality-Aware Gated Network for Multimodal MRI-Based Brain Tumor Segmentation"**.

## Contribution

MoGateNet focuses on improving multimodal MRI-based brain tumor segmentation by addressing the limitation of conventional multimodal fusion strategies. Existing methods commonly concatenate multiple MRI modalities or fuse them uniformly, which may fail to fully exploit modality-specific information across heterogeneous tumor subregions.

To address this issue, we propose a modality-aware 3D segmentation network that enhances feature fusion at both the bottleneck and skip-connection levels. MoGateNet consists of four modality-specific encoder branches, Modality-Aware Spatial Attention (MASA), NMaFA-based bottleneck refinement, Modality Gate-based skip refinement, and a decoder with deep supervision.

Extensive experiments on the BraTS 2020 dataset and external validation on the UCSF-PDGM dataset demonstrate the effectiveness and generalization ability of MoGateNet.

## Architecture

<p align="center">
  <img src="asset/overall.png" width="850">
</p>

## Dependencies

First, please make sure you have installed Python and pip. Then, the required dependencies can be installed by:

```bash
pip install -r requirements.txt
```

## Data Preparation

Please download the BraTS 2020 and UCSF-PDGM datasets from their official sources. Each case should include four MRI modalities, FLAIR, T1, T1ce, and T2, along with the corresponding segmentation label.

Due to dataset redistribution restrictions, the datasets are not included in this repository.

## Training

After the data is prepared, you can train the model using the following command:

```bash
python main.py
```


## Evaluation

After training is completed, the best model checkpoint is saved in the specified log directory.

The evaluation reports Dice and HD95 scores for TC, WT, and ET.

## Experiment Results

Here are the main results of our experiment:

<p align="center">
  <img src="asset/result1.png" width="850">
</p>

Also, external validation results on the UCSF-PDGM dataset are shown below:

<p align="center">
  <img src="asset/result2.png" width="850">
</p>


