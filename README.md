# ReManNet: A Riemannian Manifold Network for Monocular 3D Lane Detection

<p align="center">
  <b>ReManNet: A Riemannian Manifold Network for Monocular 3D Lane Detection</b>
</p>

<p align="center">
  <a href="./paper/ReManNet.pdf">
    <img src="https://img.shields.io/badge/Paper-PDF-red" alt="Paper">
  </a>
  <a href="https://github.com/changehome717/ReManNet">
    <img src="https://img.shields.io/badge/Code-ReManNet-blue" alt="Code">
  </a>
  <img src="https://img.shields.io/badge/Task-Monocular%203D%20Lane%20Detection-green" alt="Task">
  <img src="https://img.shields.io/badge/Framework-PyTorch-orange" alt="Framework">
</p>

<p align="center">
  Official implementation of <b>ReManNet</b>, a Riemannian manifold network for monocular 3D lane detection.
</p>

---

## News

- **[2026]** ReManNet is released for monocular 3D lane detection.
- Code, configurations, pretrained models, and visualization tools will be updated progressively.
- The paper is available at:

[ReManNet: A Riemannian Manifold Network for Monocular 3D Lane Detection](https://arxiv.org/abs/2603.19776)

---

---

## Introduction

Monocular 3D lane detection aims to recover lane geometry in 3D space from a single front-view image. It is a fundamental perception task for autonomous driving, providing metric lane structure for downstream planning, lane keeping, and scene understanding.

However, monocular 3D lane detection remains challenging due to depth ambiguity and weak geometric constraints. Existing methods usually rely on depth guidance, BEV projection, or anchor-based and curve-based prediction heads. Although effective, these methods often remap high-dimensional image features while only weakly encoding the intrinsic geometry of roads and lanes.

To address this issue, we propose **ReManNet**, a Riemannian manifold network for monocular 3D lane detection. ReManNet is built upon the **Road-Manifold Assumption**, where the road surface is modeled as a smooth 2D manifold embedded in 3D space, lane markings are regarded as 1D submanifolds, and sampled lane points are treated as dense observations.

Based on this formulation, ReManNet encodes lane geometry as Riemannian Gaussian descriptors on the symmetric positive-definite manifold and fuses them with visual features through a lightweight gating module. We also introduce the **3D Tunnel Lane IoU Loss**, which provides shape-level supervision by computing slice-wise overlap between tubular neighborhoods along predicted and ground-truth lanes.

---

## Framework

<p align="center">
  <img src="assets/framework.png" width="900">
</p>

The overall pipeline of ReManNet contains the following stages:

1. A monocular image is processed by an image backbone and detection head to generate initial 3D lane predictions.
2. The predicted lane points are fed into a position-weighted geometric encoder.
3. Local lane geometry is summarized as Riemannian Gaussian descriptors on the SPD manifold.
4. Manifold descriptors are mapped to compact Euclidean features through logarithmic mapping and projection.
5. A gated visual-geometric fusion module combines visual features and geometric descriptors.
6. The final predictions are supervised by classification, coordinate regression, and the proposed 3D-TLIoU loss.

---

## Highlights

### Road-Manifold Assumption

We introduce the Road-Manifold Assumption to provide a principled geometric foundation for monocular 3D lane detection.

The assumption models:

- road surfaces as smooth 2D manifolds in 3D space,
- lane markings as embedded 1D submanifolds,
- sampled lane points as dense observations of lane curves.

This formulation couples metric and topological structure across road surfaces, lane curves, and point sets.

### Riemannian Gaussian Geometry Encoding

ReManNet encodes local lane geometry as Riemannian Gaussian descriptors on the symmetric positive-definite manifold. This representation captures spatial correlations, local structural consistency, and geometric variation of lane points.

### Gated Visual-Geometric Fusion

A lightweight gate adaptively fuses visual features and manifold-based geometric descriptors. The visual stream provides image evidence, while the geometric stream supplies structure-aware correction for coherent 3D reasoning.

## 3D Tunnel Lane IoU Loss

We currently release the implementation of **3D Tunnel Lane IoU Loss (3D-TLIoU)**, a plug-and-play geometry-consistent supervision term for ordered 3D lane point sequences.

Different from conventional point-wise regression losses that independently penalize each sampled point, 3D-TLIoU treats a lane as a continuous 3D curve and evaluates the shape-level consistency between predicted and ground-truth lane sequences. For each sampled longitudinal position, it constructs local tunnel neighborhoods around the predicted and ground-truth lane points and computes a slice-wise overlap surrogate. The loss then aggregates these slice-wise geometric overlaps together with local directional consistency, encouraging both accurate point localization and coherent curve alignment.

<p align="center">
  <img src="assets/3dtliou.png" width="800">
</p>

<p align="center">
  <em>Illustration of the proposed 3D-TLIoU loss. Please place the schematic figure at <code>assets/3dtliou.png</code>.</em>
</p>

Given a predicted 3D lane and its corresponding ground truth, 3D-TLIoU measures the discrepancy from two complementary perspectives:

- **Tunnel overlap consistency**, which evaluates whether predicted lane points remain within the local 3D neighborhood of the ground-truth lane.
- **Directional consistency**, which regularizes the local tangent direction of the predicted lane curve.

The slice-wise overlap term is formulated as a signed tunnel-overlap surrogate. When the predicted and ground-truth tunnel neighborhoods overlap, the term behaves similarly to an IoU-style alignment objective. When they are disjoint, the signed overlap is allowed to become negative, which explicitly reflects the separation between the two tunnel regions. This design provides a stronger distance-aware penalty for predictions that move far away from the ground-truth lane, instead of treating all non-overlapping cases equally.

This makes 3D-TLIoU especially suitable for supervising ordered 3D lane points, where maintaining global curve coherence is as important as minimizing local coordinate errors.

---

## Main Results

### OpenLane

| Method | Backbone | F1 (%) ↑ | Cate Acc (%) ↑ | Ex/N ↓ | Ex/F ↓ | Ez/N ↓ | Ez/F ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| Anchor3DLane | ResNet-50 | 57.5 | 91.6 | 0.233 | 0.246 | 0.080 | 0.106 |
| ReManNet | ResNet-18 | 63.5 | 92.8 | 0.222 | 0.265 | 0.069 | 0.089 |
| ReManNet | ResNet-50 | **65.7** | **94.7** | **0.189** | **0.205** | **0.060** | **0.072** |

### Scenario-Level Improvements on OpenLane

ReManNet achieves strong performance in challenging scenarios, including:

- uphill and downhill roads,
- curved roads,
- extreme weather,
- nighttime scenes,
- intersections,
- merge and split scenarios.

Compared with the baseline, ReManNet improves the OpenLane F1 score by **+8.2%**, and surpasses the previous best method by **+1.8%**.

---

<!--
## Installation

### Environment

This project is implemented with PyTorch. We recommend using a conda environment.

```bash
conda create -n remannet python=3.8 -y
conda activate remannet
```

Install PyTorch according to your CUDA version. For example:

```bash
pip install torch torchvision torchaudio
```

Then install the required packages:

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

### OpenLane

Please download the OpenLane dataset from the official source and organize it as follows:

```text
data/
└── OpenLane/
    ├── images/
    ├── lane3d_300/
    ├── training/
    └── validation/
```

### ApolloSim

Please download ApolloSim and organize it as follows:

```text
data/
└── ApolloSim/
    ├── images/
    ├── labels/
    └── splits/
```

Please modify the dataset paths in the configuration files before training or evaluation.

---

## Training

To train ReManNet on OpenLane with a ResNet-50 backbone:

```bash
python tools/train.py configs/remannet/remannet_openlane_r50.py
```

To train with a ResNet-18 backbone:

```bash
python tools/train.py configs/remannet/remannet_openlane_r18.py
```

For distributed training:

```bash
bash tools/dist_train.sh configs/remannet/remannet_openlane_r50.py 4
```

---

## Evaluation

To evaluate a trained model:

```bash
python tools/test.py configs/remannet/remannet_openlane_r50.py \
    checkpoints/remannet_openlane_r50.pth
```

For distributed evaluation:

```bash
bash tools/dist_test.sh configs/remannet/remannet_openlane_r50.py \
    checkpoints/remannet_openlane_r50.pth 4
```

---
-->

## Current Release

We currently release the implementation of **3D Tunnel Lane IoU Loss (3D-TLIoU)** for ordered 3D lane point supervision.

The released loss can be easily integrated into existing 3D lane detection or 3D point-sequence regression frameworks. It provides shape-level supervision by jointly considering point-wise proximity and local directional consistency along the lane sequence. Compared with conventional independent point-wise regression losses, 3D-TLIoU encourages more coherent curve-level alignment and can improve the supervision quality of 3D sequential points with minimal modification to existing training pipelines.

The full ReManNet codebase, including the Riemannian manifold geometry modules, visual-geometric fusion components, training configurations, evaluation scripts, and pretrained models, will be released progressively.

Please consider **starring** and **watching** this repository to receive updates when the complete implementation is available.

---

## Released Components

| Component | Status | Description |
|---|---|---|
| 3D-TLIoU Loss | Released | Geometry-consistent supervision for ordered 3D lane point sequences |
| Riemannian Gaussian Geometry Module | Coming soon | SPD-manifold-based geometric descriptor encoding |
| Gated Visual-Geometric Fusion | Coming soon | Adaptive fusion of visual and manifold geometric features |
| Training and Evaluation Code | Coming soon | Full pipeline for OpenLane and ApolloSim |
| Pretrained Models | Coming soon | Checkpoints for reproduced benchmark results |

---

<!--
## Visualization

<p align="center">
  <img src="assets/visualization.png" width="900">
</p>

ReManNet improves 3D lane structure prediction by reducing geometric distortion such as concavity, bulging, twisting, and far-range localization drift.

---
-->

<!--
## Code Structure

ReManNet/
├── configs/                  # Configuration files
├── data/                     # Dataset directory
├── docs/                     # Additional documentation
├── models/                   # Model definitions
│   ├── backbones/            # Image backbones
│   ├── heads/                # Detection heads
│   ├── geometry/             # Riemannian geometry modules
│   └── losses/               # Loss functions, including 3D-TLIoU
├── tools/                    # Training and evaluation scripts
├── assets/                   # Figures and visualization examples
├── checkpoints/              # Pretrained models
├── requirements.txt
├── LICENSE
└── README.md

---
-->

<!--
## TODO

- [ ] Release full training code.
- [ ] Release evaluation code.
- [ ] Release pretrained models.
- [ ] Release OpenLane configuration files.
- [ ] Release ApolloSim configuration files.
- [ ] Add visualization scripts.
- [ ] Add detailed documentation for the Riemannian geometry module.

---
-->

## Citation

If you find this project useful for your research, please consider starring this repository and citing our work:

```bibtex
@inproceedings{hong2026remannet,
  title     = {ReManNet: A Riemannian Manifold Network for Monocular 3D Lane Detection},
  author    = {Hong, Chengzhi and Li, Bijun},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2026}
}
```

---

## Contact

For questions, discussions, or collaboration, please contact:

```text
Chengzhi Hong
State Key Laboratory of Information Engineering in Surveying, Mapping and Remote Sensing
Wuhan University
Email: ht005305@whu.edu.cn
```
