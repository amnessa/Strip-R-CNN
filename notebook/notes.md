# Evaluation notebooks (DOTA / FAIR1M / HRSC2016)

There are three workflow notebooks in this folder, one per dataset:

- `dota_strip_rcnn_s_workflow.ipynb`   — checkpoint `weights/strip_rcnn_s_dota.pth`
- `fair1m_strip_rcnn_s_workflow.ipynb` — checkpoint `weights/strip_rcnn_s_fair1m.pth`
- `hrsc2016_strip_rcnn_s_workflow.ipynb` — needs an HRSC checkpoint you supply (none ships here)

Each notebook does the same three things: load the detector, render detections
for chosen classes, and **evaluate the labelled split with mAP**. They are
generated/maintained by `_build_notebooks.py` (FAIR1M + HRSC are generated, DOTA
is patched in place to preserve existing outputs).

## Why we evaluate the validation split, not test

The official `test` splits of DOTA and FAIR1M ship **without public labels** —
they are scored only by the online evaluation servers. So the only honest local
metric comes from a split whose ground truth we hold:

- **DOTA**: evaluate the `val` split (`EVAL_SPLIT='val'`, `METRIC='mAP'`).
- **FAIR1M**: the labelled `train` split is exposed as `val` by `fairv1.py`; that
  is what we score (`EVAL_SPLIT='val'`).
- **HRSC2016**: the local `test` split *is* labelled (XML annotations), so we
  score it directly (`EVAL_SPLIT='test'`).

## GPU / sm_120 situation (the original blocker)

The box has an RTX 5070 Ti (Blackwell, **sm_120**). The notebooks are now
**GPU-first**: `pick_device()` returns `cuda:0` whenever the running torch build
advertises sm_120, and there is a `FORCE_DEVICE` override at the top of the
config cell (`'cuda:0'` / `'cpu'` / `None` for auto). On GPU the eval cells
default to the full split (`MAX_SUBSET_IMAGES = None`); on CPU they bound it
because Strip R-CNN is a few minutes per image on CPU.

The catch is the environment. Across the conda envs:

| env        | torch          | sm_120 | mmcv/mmdet/mmrotate |
|------------|----------------|--------|---------------------|
| openmmlab  | 1.8.0 / cu10.2 | no     | 1.7.2 / 2.28 / 0.3.4 (this codebase) |
| sdfnet     | 2.8.0 / cu12.8 | **yes**| missing             |
| segdiff    | 1.9.0 / cu11.1 | no     | missing             |

So `openmmlab` has the right libraries but a torch that cannot drive the GPU,
and `sdfnet` has a torch that *can* drive the GPU but none of the OpenMMLab
stack. sm_120 needs CUDA 12.8 / torch 2.7+, and **mmcv 1.x has no prebuilt
wheels for that** — enabling the GPU means building `mmcv-full` 1.7.2 from
source against torch 2.8 (cu128) in an sm_120 env, then installing `mmdet`
2.28.2 and this repo (`pip install -e .`). That build can need source patches
(torch 2.x C++ API changes) and is not guaranteed first try. The alternative —
porting StripNet/StripHead to the OpenMMLab 2.x stack (mmcv 2.x + mmdet 3.x +
mmrotate 1.x), which *does* have cu128 wheels — is a larger code change.

### How to enable the GPU yourself (when ready)

Build the OpenMMLab 1.x stack against the sm_120 torch, in a **clone** so the
working `sdfnet`/`openmmlab` envs stay intact:

```bash
# 1. Clone the sm_120 torch env (torch 2.8.0+cu128 already supports sm_120)
conda create -n strip-gpu --clone sdfnet
conda activate strip-gpu
python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"
# expect: 2.8.0+cu128 [... 'sm_120']

# 2. Build mmcv-full 1.7.2 from source against this torch (no cu128 wheel exists).
#    MMCV_WITH_OPS=1 compiles the CUDA ops; TORCH_CUDA_ARCH_LIST targets Blackwell.
pip install -r <(echo "ninja psutil")
MMCV_WITH_OPS=1 TORCH_CUDA_ARCH_LIST="12.0" \
  pip install "mmcv-full==1.7.2" --no-binary mmcv-full -v
#    If the compile fails on torch 2.x C++ API changes, that is the known risk;
#    capture the first error and patch the offending op or pin a newer mmcv 1.x commit.

# 3. Install the detector stack + this repo
pip install "mmdet==2.28.2"
cd /home/cagolinux/Strip-R-CNN && pip install -e .

# 4. Sanity check the ops actually run on the GPU
python -c "import torch, mmcv.ops as o; \
b=torch.tensor([[0,0,10,10,0]],dtype=torch.float32,device='cuda'); \
print('nms_rotated ok:', o.nms_rotated(b, torch.tensor([0.9],device='cuda'), 0.1)[0].shape)"
```

Then launch Jupyter from `strip-gpu` and open any of the three notebooks. With a
sm_120-capable torch, `pick_device()` prints `Using GPU: ...` and the whole
pipeline (inference + `dataset.evaluate(metric='mAP')`) runs on the GPU — set
`MAX_SUBSET_IMAGES = None` to score the full validation split. Nothing in the
notebooks needs editing; they detect the GPU automatically. If a step in (2)
fails, the fallback is the OpenMMLab 2.x port noted above.

## DOTA validation data layout (what was fixed)

The DOTA data on disk is the **original** validation set, not MMRotate split
tiles. `mmrotate/data/DOTA/`:

- `validation/images/` — 458 full scenes (~1000px each)
- `validation/labelTxt-.../labelTxt-v1.0/labelTxt.zip` — original OBB labels,
  which include `imagesource:`/`gsd:` header lines that `DOTADataset` cannot parse.

Fix applied: the labels were extracted and cleaned (header lines stripped) into
`validation/.../labelTxt-v1.0/_extracted` → `val/annfiles_clean`, and a clean
`val/` layout was created via symlinks so the standard MMRotate paths work:

- `mmrotate/data/DOTA/val/images`   → `../validation/images`
- `mmrotate/data/DOTA/val/annfiles` → `annfiles_clean` (458 files, headers removed)

Verified: building the val dataset gives 456 usable images, and
`dataset.evaluate(perfect_preds, metric='mAP')` returns 1.0 — confirmed both in a
guarded script and inside a real Jupyter kernel (the mAP code uses
`get_context('spawn').Pool`, which works in-notebook because its target is an
importable function). For a *faithful* DOTA number you'd normally split val into
1024px tiles with `tools/data/dota/split` first; direct full-scene eval is a
reasonable baseline but not identical to the paper protocol.

## HRSC2016 still needs setup

No HRSC detector checkpoint ships here (the repo only has the DOTA and FAIR1M
detectors plus the ImageNet backbone). HRSC is single-class (`num_classes=1`), so
a DOTA/FAIR checkpoint will not load. Also, the HRSC2016 download under
`mmrotate/data/hrsc/archive` is not yet extracted into the expected
`ImageSets/` + `FullDataSet/{AllImages,Annotations}` layout. The notebook raises
clear errors until both are provided.

---

This repo is a standard MMRotate project. The actual inference entry points are image_demo.py, huge_image_demo.py, test.py, and the existing notebook MMRotate_Tutorial.ipynb. One important detail: the README has a few inconsistencies, so for real usage you should trust the config files and dataset classes first.

**Weights**

There are two different kinds of pretrained weights in this repo.

- Full detector checkpoints: these are the model downloads listed in README.md. Use these for inference or evaluation. They do not need to live in a special folder. You can keep them anywhere and pass the path as the `checkpoint` argument to `init_detector(...)` or test.py.
- Backbone-only ImageNet pretraining: the Strip-R-CNN configs hardcode these as `model.backbone.init_cfg.checkpoint="pretrained/stripnet_s.pth.tar"` in files like strip_rcnn_s_fpn_1x_dota_le90.py, strip_rcnn_s_fpn_1x_fair_le90.py, strip_rcnn_s_fpn_1x_dior_le90.py, and strip_rcnn_s_fpn_3x_hrsc_le90.py. If you want to train or fine-tune, create a repo-local `pretrained/` folder and put the backbone file there, or edit the config to point somewhere else.
- The Tiny config in strip_rcnn_t_fpn_1x_dota_le90.py currently also points to `pretrained/stripnet_s.pth.tar`. If you really use the Tiny backbone pretrain, you should check and likely change that path.

So the short answer is: you do not “install” weights. You download them and point the code at them.

**Datasets**

If you want proper MMRotate dataset evaluation or training, the dataset must match the layout expected by the selected base config.

- DOTA: dotav1.py expects split data under `trainval/annfiles`, `trainval/images`, `val/annfiles`, `val/images`, and `test/images`. The split procedure is documented in README.md.
- FAIR1M: fairv1.py expects `data/fair1_0_to_DOTA_split/train/annfiles`, `train/images`, and `test/images`.
- HRSC: hrsc.py expects `data/HRSC2016/` with `ImageSets` and `FullDataSet/...`. The doc in README.md says `data/hrsc/`, so here the config is the source of truth and you should adjust `data_root` if needed.
- DIOR: dior.py uses a machine-specific absolute path. You will need to edit `data_root`.

The repo-wide recommendation is in README.md: keep datasets under `data/` or symlink them there.

If your goal is only notebook inference and plotting, you do not need to place images into a formal dataset tree at all. You can keep images anywhere and call `inference_detector(model, image_path)` directly. The strict dataset layout only matters for dataset-wide evaluation or training.

**How to use pretrained checkpoints**

For a single image, this repo already does:

- build model with `init_detector(config, checkpoint, device=...)`
- run `inference_detector(model, img)`
- render with `show_result_pyplot(...)`

That is exactly what image_demo.py does.

For very large aerial images, use patch-based inference from huge_image_demo.py, which calls `inference_detector_by_patches(...)` from inference.py.

For dataset-wide evaluation or visualization, use test.py, for example:

```bash
python tools/test.py \
  configs/strip_rcnn/strip_rcnn_s_fpn_1x_dota_le90.py \
  /path/to/detector_checkpoint.pth \
  --eval mAP \
  --show-dir work_dirs/vis
```

That path only works cleanly after you set the dataset root in the matching base dataset config.

**Notebook pattern**

For a notebook, the cleanest path is to load a detector checkpoint, run inference image-by-image, filter the class outputs you care about, and render with `model.show_result(...)`. The rendering behavior comes from base.py.

```python
from pathlib import Path

import matplotlib.pyplot as plt
import mmcv
from mmcv import Config
from mmdet.apis import init_detector, inference_detector
import mmrotate  # registers mmrotate modules

config_path = "configs/strip_rcnn/strip_rcnn_s_fpn_1x_dota_le90.py"
checkpoint_path = "/path/to/strip_rcnn_s_dota_checkpoint.pth"

cfg = Config.fromfile(config_path)

# If you are loading a full detector checkpoint, this avoids needing the
# backbone-only pretrained file during model construction.
cfg.model.backbone.init_cfg = None

model = init_detector(cfg, checkpoint_path, device="cuda:0")
print(model.CLASSES)

img_path = "/path/to/one/image.png"
result = inference_detector(model, img_path)

# Keep only the classes you want.
# For DOTA there is no literal "car"; the closest classes are
# "small-vehicle" and "large-vehicle".
wanted = {"ship", "small-vehicle", "large-vehicle"}

filtered_result = [
    cls_result if class_name in wanted else cls_result[:0]
    for class_name, cls_result in zip(model.CLASSES, result)
]

vis = model.show_result(
    img_path,
    filtered_result,
    score_thr=0.3,
    show=False
)

plt.figure(figsize=(12, 12))
plt.imshow(mmcv.bgr2rgb(vis))
plt.axis("off")
plt.show()
```

To run that over a folder of images:

```python
image_dir = Path("/path/to/images")

for img_path in sorted(image_dir.glob("*.png"))[:10]:
    result = inference_detector(model, str(img_path))
    filtered_result = [
        cls_result if class_name in wanted else cls_result[:0]
        for class_name, cls_result in zip(model.CLASSES, result)
    ]
    vis = model.show_result(str(img_path), filtered_result, score_thr=0.3, show=False)

    plt.figure(figsize=(10, 10))
    plt.title(img_path.name)
    plt.imshow(mmcv.bgr2rgb(vis))
    plt.axis("off")
    plt.show()
```

If your images are huge remote-sensing scenes, replace `inference_detector(...)` with the patch API:

```python
from mmrotate.apis import inference_detector_by_patches

result = inference_detector_by_patches(
    model,
    img_path,
    sizes=[1024],
    steps=[824],
    ratios=[1.0],
    merge_iou_thr=0.1
)
```

**About `train`, `car`, and `ship`**

This is the main limitation: none of the pretrained Strip-R-CNN checkpoints in this repo appears to use the exact label set `train`, `car`, `ship`.

- DOTA classes in dota.py include `ship`, `small-vehicle`, and `large-vehicle`, but not `train`.
- FAIR classes in fair.py include many ship and vehicle subclasses such as `Passenger_Ship`, `Small_Car`, `Cargo_Truck`, etc., but still not `train`.
- DIOR classes in dior.py include `ship`, `vehicle`, and `trainstation`. `trainstation` is not the same as a train object.
- HRSC is effectively ship-only.

So:

- If you use the DOTA checkpoint, filter `ship`, `small-vehicle`, `large-vehicle`.
- If you use the FAIR checkpoint, filter the ship and vehicle subclasses you want.
- If you truly need a literal `train` class, you will need a checkpoint trained on a dataset that actually contains `train`, or you will need to fine-tune one.

If you want, I can next give you a notebook template tailored to one specific checkpoint and dataset pair, for example:
1. DOTA + Strip R-CNN-S
2. FAIR1M + Strip R-CNN-S
3. DIOR + Strip R-CNN-S