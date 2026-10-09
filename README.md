# PRATH

**A Training-Efficient Pose Regression Network With Anchor-Based Transformer and Heatmap Branch**

Official PyTorch implementation of PRATH — a dual-branch human pose estimation framework that
integrates an **anchor-based Transformer regression branch** with an **auxiliary heatmap branch**
for efficient and accurate end-to-end keypoint regression.

> Paper: *A Training-Efficient Pose Regression Network With Anchor-Based Transformer and Heatmap Branch* —
> Zhigang Yang, Qian Miao, Xinbo Jia, Zehao Gao, Lutao Liu.

---

## Highlights

- **Dual-branch design.** A regression branch predicts keypoints end-to-end, while an auxiliary
  heatmap branch provides denser intermediate supervision on the backbone features.
- **Anchor points as query priors.** Instead of letting each query attend over the whole image,
  anchor points are integrated with the queries to *limit the scope of the query*. This
  significantly accelerates the convergence of training and improves prediction accuracy.
- **Zero extra inference cost.** The heatmap branch is employed **only during training** and is
  discarded at inference time, so it adds no runtime overhead.
- **Multi-level feature fusion.** Multi-level feature maps produced by the backbone are fused to
  enrich both semantic and location information before the Transformer.
- **Better accuracy / efficiency trade-off.** PRATH reaches competitive accuracy against
  heatmap-based, classification-based and regression-based methods, with notably strong
  `AP50` performance and fewer training epochs.

---

## Installation

```bash
conda create -n prath python=3.8 -y
conda activate prath

# PyTorch (match your CUDA version)
pip install torch torchvision

# dependencies
cd PRATH-main/two_stage
pip install -r requirements.txt

# required by lib/models/anchor_detr.py and lib/models/transformer.py
# (not listed in requirements.txt of the upstream baseline)
pip install bottleneck-transformer-pytorch
```

PRATH is implemented in pure PyTorch and does **not** require compiling CUDA extensions.

> Verified environment: 2 × NVIDIA GeForce RTX 3060. Training was performed with AdamW,
> base learning rate `1e-4` for the backbone and `1e-5` for the Transformer, halved at the
> 60th and 70th epochs, and loss weights `δ_c = 1`, `δ_r = 5`, `δ_h = 2`.

---

## Data preparation

Download the public benchmarks and make them available to the configs:

- **COCO 2017** — <https://cocodataset.org> (about 250K person instances, 17 keypoints).
  Training uses `train2017` (~57K images) with `person_keypoints` annotations.
- **MPII Human Pose** — <http://human-pose.mpi-inf.mpg.de>

Either set `DATASET.ROOT` inside the experiment YAML, or pass the dataset root at runtime with
`--dataDir` (the same applies to `--logDir` and `--modelDir`; all three default to `''`, meaning
"take the value from the config file").

---

## Training

Run from `PRATH-main/two_stage`:

```bash
# MPII, ResNet-50
python tools/train.py --cfg experiments/mpii/transformer/res50_384x384_adamw_lr1e-4.yaml

# COCO, HRNet-W32
python tools/train.py --cfg experiments/coco/transformer/w32_384x288_adamw_lr1e-4.yaml
```

Override any config entry on the command line (config keys are appended as `key value` pairs):

```bash
python tools/train.py --cfg experiments/mpii/transformer/res50_384x384_adamw_lr1e-4.yaml \
    --dataDir /path/to/data --logDir log --modelDir model
```

Available configs:

| Dataset | Directory | Backbones |
|---|---|---|
| COCO | `experiments/coco/transformer/` | ResNet-50, ResNet-101, HRNet-W32 |
| COCO | `experiments/coco/deformable_transformer/` | ResNet-50, ResNet-101 |
| MPII | `experiments/mpii/transformer/` | ResNet-50, ResNet-101, ResNet-152, HRNet-W32 |

---

## Evaluation

```bash
python tools/test.py --cfg experiments/mpii/transformer/res50_384x384_adamw_lr1e-4.yaml \
    --modelDir <CKPT_DIR> \
    TEST.MODEL_FILE <CKPT_FILE>
```

---

## Pretrained models

Pretrained checkpoints are **not** included in this repository (training logs and `.pth` files are
excluded by `.gitignore`). They will be released together with the camera-ready version of the paper.

---

## License

This project is released under the [Apache License 2.0](PRATH-main/LICENSE), consistent with the licenses of the
upstream projects this work builds upon.
