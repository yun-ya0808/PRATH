# PRATH

**A Training-Efficient Pose Regression Network With Anchor-Based Transformer and Heatmap Branch**

Official PyTorch implementation of PRATH — a dual-branch human pose estimation framework that  
integrates an **anchor-based Transformer regression branch** with an **auxiliary heatmap branch**  
for efficient and accurate end-to-end keypoint regression.

> Paper: *A Training-Efficient Pose Regression Network With Anchor-Based Transformer and Heatmap Branch* —  
> Zhigang Yang, Qian Miao, Xinbo Jia, Zehao Gao, Lutao Liu.  
> Submitted to *Image and Vision Computing*.

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

## Results

### COCO `val` set

| Method | Backbone      | Input size | Params (M) | GFLOPs | AP       | AP50     | AP75     | APM      | APL      | AR       |
| ------ | ------------- | ---------- | ---------- | ------ | -------- | -------- | -------- | -------- | -------- | -------- |
| PRATH  | MobileNet-v2  | 384×288    | 19.4       | 6.7    | 70.2     | 89.0     | 77.6     | 67.1     | 77.1     | 76.8     |
| PRATH  | ResNet-50     | 384×288    | 41.6       | 11.0   | 71.4     | 90.7     | 78.5     | 69.2     | 77.8     | 77.4     |
| PRATH  | ResNet-101    | 384×288    | 60.5       | 19.1   | 73.3     | 91.7     | 80.6     | 70.5     | 79.4     | 79.6     |
| PRATH  | **HRNet-W32** | 384×288    | 57.3       | 21.6   | **74.6** | **91.9** | **81.1** | **71.6** | **80.1** | **80.2** |

For reference, the regression-based baseline PRTR (HRNet-W32, 384×288) obtains  
AP 73.1 / AP50 89.4 / AP75 79.8 / APM 68.8 / APL 80.4 / AR 79.8.

### MPII `val` set (PCKh@0.5)

| Method | Backbone      | Head     | Shoulder | Elbow | Wrist | Hip  | Knee | Ankle | Mean     |
| ------ | ------------- | -------- | -------- | ----- | ----- | ---- | ---- | ----- | -------- |
| PRATH  | ResNet-50     | 95.9     | 94.0     | 87.0  | 80.1  | 87.5 | 81.4 | 75.3  | 86.6     |
| PRATH  | ResNet-101    | 96.4     | 95.0     | 87.9  | 89.2  | 88.6 | 83.1 | 77.4  | 88.1     |
| PRATH  | **HRNet-W32** | **97.3** | 95.7     | 89.4  | 89.8  | 89.1 | 85.7 | 78.9  | **89.3** |

Ablation studies (heatmap-branch variants, heatmap resolution, and loss-weight configurations)  
are reported in the paper.

---

## Repository structure

```
PRATH/
└── PRATH-main/
    ├── two_stage/                     # main two-stage implementation (used in the paper)
    │   ├── experiments/
    │   │   ├── coco/transformer/      # COCO configs (res50 / res101 / w32, AdamW, lr 1e-4)
    │   │   ├── coco/deformable_transformer/
    │   │   └── mpii/transformer/      # MPII configs (res50 / res101 / res152 / w32)
    │   ├── lib/
    │   │   ├── core/                  # function.py (train loop), loss.py, evaluate.py, inference.py
    │   │   ├── dataset/               # coco.py, mpii.py, JointsDataset.py
    │   │   ├── models/                # anchor_detr.py ★, transformer.py, backbone.py, hrnet.py,
    │   │   │                          # matcher.py, positional_encoding.py, row_column_decoupled_attention.py
    │   │   └── utils/
    │   ├── tools/                     # train.py, test.py, trace.py
    │   ├── requirements.txt
    │   └── README.md                  # original two-stage README of the baseline
    ├── sequential/                    # single-stage variant
    └── figures/
```

The core of the contribution lives in `two_stage/lib/models/anchor_detr.py`,  
`two_stage/lib/core/loss.py` and `two_stage/lib/core/function.py`.

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

| Dataset | Directory                                  | Backbones                                    |
| ------- | ------------------------------------------ | -------------------------------------------- |
| COCO    | `experiments/coco/transformer/`            | ResNet-50, ResNet-101, HRNet-W32             |
| COCO    | `experiments/coco/deformable_transformer/` | ResNet-50, ResNet-101                        |
| MPII    | `experiments/mpii/transformer/`            | ResNet-50, ResNet-101, ResNet-152, HRNet-W32 |

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

## Citation

If you find this work useful, please cite:

```bibtex
@article{yang2026prath,
  title   = {A Training-Efficient Pose Regression Network With Anchor-Based Transformer and Heatmap Branch},
  author  = {Yang, Zhigang and Miao, Qian and Jia, Xinbo and Gao, Zehao and Liu, Lutao},
  journal = {Image and Vision Computing},
  year    = {2026},
  note    = {Under review}
}
```

---

## Acknowledgement

This implementation is built on top of  
[**PRTR: Pose Recognition with Cascade Transformers**](https://github.com/mlpc-ucsd/PRTR)  
(CVPR 2021). Parts of `lib/models` are further adapted from  
[Anchor-DETR](https://github.com/megvii-research/AnchorDETR),  
[Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) and  
[DETR](https://github.com/facebookresearch/detr).  
Backbone implementations follow [HRNet](https://github.com/HRNet/HRNet-Human-Pose-Estimation).  
We thank the authors for making their code publicly available.

This work was supported in part by the National Natural Science Foundation of China under Grant  
61201238, and in part by the Aeronautical Science Foundation of China under Grant 201801P6002.

---

## License

This project is released under the [Apache License 2.0](LICENSE), consistent with the licenses of  
the upstream projects listed above.
