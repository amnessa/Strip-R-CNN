

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