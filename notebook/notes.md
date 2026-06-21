Searched codebase for "please investigate this project and explain to me how can I use pretrained models and weights belonging to this project. where should I install them? how can I use pretrained weights to do object detection on datasets? where should I put the datasets? my aim is to create a notebook that utilizes pretrained models on datasets and renders the detection results as a plot on screen. I also want to filter classes and only render bounding boxes for train, car and ship classes", no results

I’m checking the repo surfaces that actually control pretrained checkpoints, dataset layout, and inference so I can answer with concrete paths and notebook-ready guidance.

Searched for regex `pretrained|checkpoint|load_from|classes|CLASSES|data_root|ann_file|img_prefix` (`**/README.md|configs/**|demo/**|docs/en/**|mmrotate/**`), no results

Searched for files matching `**/configs/strip_rcnn/*.py`, 12 matches

Searched for files matching `**/demo/*.py`, 6 matches

Read memory

I’ve isolated the Strip R-CNN configs and demo entry points; next I’m reading those plus the dataset definitions so the usage advice matches how this repo actually expects checkpoints and annotations.

Searched for regex `load_from|resume_from|pretrained` (`**/configs/strip_rcnn/**`), 24 results

Searched for regex `classes|CLASSES` (`**/mmrotate/datasets/**`), 51 results

Searched for regex `data_root|ann_file|img_prefix` (`**/configs/strip_rcnn/**`), no results

Searched for regex `init_detector|inference_detector|show_result_pyplot|show_result` (`**/demo/**|mmrotate/apis/**|tools/**`), no results

The repo’s own configs point backbone pretraining to a local `pretrained/stripnet_s.pth.tar`, so now I’m tracing the exact inference APIs and dataset folder layouts those configs inherit.

Searched for regex `init_detector|inference_detector|show_result_pyplot|show_result` (`**/demo/**`), 46 results

Searched for regex `init_detector|inference_detector|show_result_pyplot|show_result` (`**/mmrotate/apis/**`), 3 results

Searched for regex `data_root|ann_file|img_prefix` (`**/configs/_base_/datasets/**`), 200 results

Searched for regex `StripNet|stripnet` (`**/mmrotate/models/**|configs/strip_rcnn/**`), no results

I have the key anchors now: demo scripts for inference, base dataset configs for folder layout, and Strip-R-CNN configs for how backbone pretraining is wired. I’m reading those files directly so I can map “which weight goes where” without guessing.

Read [](file:///home/cago/Strip-R-CNN/demo/image_demo.py)

Read [](file:///home/cago/Strip-R-CNN/demo/huge_image_demo.py)

Read [](file:///home/cago/Strip-R-CNN/configs/strip_rcnn/strip_rcnn_s_fpn_1x_dota_le90.py)

Read [](file:///home/cago/Strip-R-CNN/configs/_base_/datasets/dotav1.py)

I need one more pass over the dataset classes and the remaining Strip-R-CNN configs, because your class filter request only makes sense if the chosen checkpoint was trained on a dataset that actually contains those labels.

Read [](file:///home/cago/Strip-R-CNN/mmrotate/datasets/dota.py#1-1), lines 1 to 80

Read [](file:///home/cago/Strip-R-CNN/mmrotate/datasets/fair.py#1-1), lines 1 to 90

Read [](file:///home/cago/Strip-R-CNN/mmrotate/datasets/dior.py#1-1), lines 1 to 80

Read [](file:///home/cago/Strip-R-CNN/configs/strip_rcnn/strip_rcnn_s_fpn_1x_fair_le90.py#1-1), lines 1 to 120

Read [](file:///home/cago/Strip-R-CNN/configs/strip_rcnn/strip_rcnn_s_fpn_1x_dior_le90.py#1-1), lines 1 to 120

Read [](file:///home/cago/Strip-R-CNN/configs/strip_rcnn/strip_rcnn_s_fpn_3x_hrsc_le90.py#1-1), lines 1 to 120

I’ve confirmed there are two different kinds of weights here: the backbone pretrain path embedded in the configs, and the full detector checkpoint passed at inference time. I’m checking the remaining dataset roots and the standard test entry point so I can separate those clearly.

Searched for regex `load_from|resume_from` (`**/configs/strip_rcnn/**`), no results

Read [](file:///home/cago/Strip-R-CNN/configs/_base_/datasets/fairv1.py)

Read [](file:///home/cago/Strip-R-CNN/configs/_base_/datasets/dior.py)

Read [](file:///home/cago/Strip-R-CNN/configs/_base_/datasets/hrsc.py)

Read [](file:///home/cago/Strip-R-CNN/tools/test.py#1-1), lines 1 to 220

I’ve got the dataset roots and the test entry point. One last read is enough to confirm how checkpoints expose class names during inference, which matters for filtering rendered labels in a notebook.

Read [](file:///home/cago/Strip-R-CNN/tools/test.py#220-220), lines 220 to 264

Searched for regex `CLASSES|checkpoint\['meta'\]|model.CLASSES` (`**/demo/**|tools/**|mmrotate/**`), no results

Searched for files matching `**/mmrotate/models/backbones/*.py`, 5 matches

Read [](file:///home/cago/Strip-R-CNN/mmrotate/apis/inference.py)

I don’t need more repo structure, but I do need the exact result shape for a clean class-filtering example. I’m checking the visualization path so the notebook snippet matches what `show_result_pyplot` expects.

Searched for regex `def show_result|show_result\(` (`**/mmrotate/models/**`), 1 result

Searched for text `show_result_pyplot` (`**/mmrotate/**|demo/**`), no results

Read [](file:///home/cago/Strip-R-CNN/mmrotate/models/detectors/base.py)

Your requested label set is the main constraint now. I’m doing a quick class-name sweep to verify whether any provided Strip-R-CNN dataset actually has a literal `train` category, versus only `vehicle` or `trainstation`.

Searched for regex `train|Train|locomotive|rail` (`**/mmrotate/datasets/**`), 6 results

Read [](file:///home/cago/Strip-R-CNN/README.md)

Read [](file:///home/cago/Strip-R-CNN/configs/strip_rcnn/strip_rcnn_t_fpn_1x_dota_le90.py#1-1), lines 1 to 60

I have enough to explain the repo’s actual behavior. I’m only pulling the official “get started” and customization docs now to see whether they already document the dataset layouts or if the configs are the only reliable source.

Searched for regex `test.py|image_demo.py|inference_detector|show_result_pyplot|data_root` (`**/docs/en/**`), 23 results

Searched for files matching `**/docs/en/get_started.md`, 1 match

Searched for files matching `**/docs/en/tutorials/customize_dataset.md`, 1 match

Read [](file:///home/cago/Strip-R-CNN/docs/en/get_started.md#1-1), lines 1 to 120

Read [](file:///home/cago/Strip-R-CNN/docs/en/tutorials/customize_config.md#1-1), lines 1 to 80

Read [](file:///home/cago/Strip-R-CNN/docs/en/install.md#70-70), lines 70 to 110

The remaining question is whether you mean “run on the benchmark test/val split” or “run on my own image collection.” I’m checking the dataset tutorial just enough to frame that distinction accurately, because the folder requirements are very different.

Searched for files matching `**/tools/data/**`, 44 matches

Read [](file:///home/cago/Strip-R-CNN/docs/en/tutorials/customize_dataset.md#1-1), lines 1 to 160

I’ve confirmed the repo expects benchmark datasets in MMRotate-style layouts, and it ships conversion/splitting utilities under data. I’m reading the short index there so I can point you to the right prep path rather than just the raw config directories.

Read [](file:///home/cago/Strip-R-CNN/tools/data/README.md)

Read [](file:///home/cago/Strip-R-CNN/tools/data/dota/README.md)

Read [](file:///home/cago/Strip-R-CNN/tools/data/hrsc/README.md)

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