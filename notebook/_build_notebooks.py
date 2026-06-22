"""Generator for the three Strip R-CNN evaluation notebooks.

Run with the openmmlab interpreter from the repo root:
    python notebook/_build_notebooks.py

It (1) writes fresh FAIR1M and HRSC2016 notebooks that mirror the proven DOTA
notebook structure, and (2) patches the existing DOTA notebook so its default
evaluation runs on the labelled *validation* split (mAP) instead of the
unlabelled test split. All three notebooks default to validation-with-labels
evaluation because the official test splits are scored only by the online
servers.
"""
from __future__ import annotations

import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


def write_nb(cells: list[dict], path: Path) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "openmmlab",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print("wrote", path)


# ---------------------------------------------------------------------------
# Shared cells
# ---------------------------------------------------------------------------

IMPORTS_CELL = """
from __future__ import annotations

import copy
import sys
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / 'configs').exists() and (candidate / 'setup.py').exists():
            return candidate
    raise FileNotFoundError('Could not find the Strip-R-CNN repository root from the current working directory.')


REPO_ROOT = find_repo_root(Path.cwd())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.parallel import MMDataParallel, scatter
from mmdet.apis import init_detector, inference_detector, single_gpu_test
from mmdet.datasets import build_dataloader

import mmrotate  # noqa: F401
from mmrotate.apis import inference_detector_by_patches
from mmrotate.datasets import build_dataset

print(f'Repository root: {REPO_ROOT}')
print(f'Torch version: {torch.__version__}')
print(f'MMRotate version: {mmrotate.__version__}')
"""

# Device picker block reused verbatim in every config cell.
DEVICE_BLOCK = """
CUDA_DEVICE_NAME = None
CUDA_CAPABILITY = None
SUPPORTED_CUDA_ARCHES = []
if torch.cuda.is_available():
    CUDA_DEVICE_NAME = torch.cuda.get_device_name(0)
    CUDA_CAPABILITY = torch.cuda.get_device_capability(0)
    SUPPORTED_CUDA_ARCHES = sorted(torch.cuda.get_arch_list())


def pick_device() -> str:
    \"\"\"Return 'cuda:0' only if this torch build can actually run on the GPU.

    The RTX 50-series (Blackwell, sm_120) is reported as available by
    torch.cuda.is_available(), but torch 1.8 was built only up to sm_75, so any
    real kernel launch fails. We detect that mismatch and fall back to CPU.\"\"\"
    if not torch.cuda.is_available():
        return 'cpu'

    current_arch = f'sm_{CUDA_CAPABILITY[0]}{CUDA_CAPABILITY[1]}'
    if current_arch not in SUPPORTED_CUDA_ARCHES:
        print(
            f'CUDA device {CUDA_DEVICE_NAME} is visible, but this torch build does not support {current_arch}. '
            f'Supported arches: {SUPPORTED_CUDA_ARCHES}. Falling back to CPU. '
            f'Rebuild PyTorch with sm_120 support to use this GPU.'
        )
        return 'cpu'

    return 'cuda:0'


DEVICE = pick_device()
"""


def eval_helpers_cell(prefix: str, kind: str) -> str:
    """Return the evaluation-helpers code cell.

    ``kind`` is 'dir' for DOTA/FAIR (annotations live in a folder) or 'imageset'
    for HRSC (annotations are selected by an ImageSets txt file).
    """
    return f"""
def subset_dataset(dataset, selected_image_ids=None, max_subset_images=None):
    selected_image_ids = [str(image_id).strip() for image_id in (selected_image_ids or []) if str(image_id).strip()]

    if not selected_image_ids and max_subset_images is None:
        return dataset

    selected_set = set(selected_image_ids)
    matched_indices = []
    matched_ids = []

    for index, info in enumerate(dataset.data_infos):
        image_id = Path(info['filename']).stem
        if not selected_set or image_id in selected_set:
            matched_indices.append(index)
            matched_ids.append(image_id)

    if selected_set:
        missing_ids = sorted(selected_set - set(matched_ids))
        if missing_ids:
            print('Warning: these image ids were not found in the selected split:', missing_ids)

    if max_subset_images is not None:
        matched_indices = matched_indices[:max_subset_images]

    if not matched_indices:
        raise ValueError('The selected subset is empty. Check SELECTED_IMAGE_IDS, MAX_SUBSET_IMAGES, and the chosen split.')

    subset = copy.deepcopy(dataset)
    subset.data_infos = [dataset.data_infos[index] for index in matched_indices]
    if hasattr(dataset, 'flag'):
        subset.flag = np.asarray(dataset.flag)[matched_indices]

    return subset


def single_device_test_cpu(model, data_loader):
    \"\"\"single_gpu_test cannot unwrap DataContainers without MMDataParallel, so
    we replicate it for CPU by scattering each batch to device -1 (= CPU).\"\"\"
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    for data in data_loader:
        with torch.no_grad():
            data = scatter(data, [-1])[0]
            result = model(return_loss=False, rescale=True, **data)

        results.extend(result)
        for _ in range(len(result)):
            prog_bar.update()

    return results


def evaluate_{prefix}_split(
    model,
    cfg: Config,
    dataset_root: Path,
    split: str = 'val',
    metric: str | None = 'mAP',
    selected_image_ids=None,
    max_subset_images=None,
    samples_per_gpu: int = 1,
    workers_per_gpu: int = 2,
):
    split = split.lower()
    if split == 'test' and metric is not None and not TEST_SPLIT_HAS_LABELS:
        raise ValueError(
            'The test split has no local labels in this layout. '
            'Use split="val" for mAP evaluation, or set metric=None.'
        )

    dataset_cfg = build_{prefix}_eval_dataset_cfg(cfg, dataset_root=dataset_root, split=split)
    dataset = build_dataset(dataset_cfg)
    dataset = subset_dataset(dataset, selected_image_ids=selected_image_ids, max_subset_images=max_subset_images)

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    base_model = model.module if hasattr(model, 'module') else model
    if next(base_model.parameters()).device.type == 'cuda':
        device_index = next(base_model.parameters()).device.index or 0
        eval_model = MMDataParallel(base_model, device_ids=[device_index])
        outputs = single_gpu_test(eval_model, data_loader, show=False)
    else:
        outputs = single_device_test_cpu(base_model, data_loader)

    metrics = None
    if metric is not None:
        metrics = dataset.evaluate(outputs, metric=metric)

    return dataset, outputs, metrics
"""


HELPERS_COMMON = """
def filter_result_by_classes(result, class_names, selected_classes):
    keep = set(validate_render_classes(selected_classes, class_names))
    filtered = []
    for class_name, class_result in zip(class_names, result):
        filtered.append(class_result if class_name in keep else class_result[:0])
    return filtered


def run_detector(model, image_path, use_patch_inference=False):
    image_path = str(image_path)
    if use_patch_inference:
        return inference_detector_by_patches(
            model, image_path, sizes=PATCH_SIZES, steps=PATCH_STEPS,
            ratios=IMG_RATIOS, merge_iou_thr=MERGE_IOU_THR,
        )
    return inference_detector(model, image_path)


def visualize_detection(model, image_path, selected_classes=None, score_thr=VIS_SCORE_THR,
                        use_patch_inference=USE_PATCH_INFERENCE, figsize=(10, 10)):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f'Image not found: {image_path}')

    selected_classes = validate_render_classes(selected_classes or RENDER_CLASSES, model.CLASSES)
    raw_result = run_detector(model, image_path, use_patch_inference=use_patch_inference)
    filtered_result = filter_result_by_classes(raw_result, model.CLASSES, selected_classes)

    rendered = model.show_result(str(image_path), filtered_result, score_thr=score_thr, show=False)

    plt.figure(figsize=figsize)
    plt.imshow(mmcv.bgr2rgb(rendered))
    plt.title(f'{image_path.name} | rendered classes: {selected_classes}')
    plt.axis('off')
    plt.show()
    return raw_result, filtered_result


def visualize_image_list(model, image_paths, selected_classes=None, score_thr=VIS_SCORE_THR,
                         use_patch_inference=USE_PATCH_INFERENCE):
    for image_path in image_paths:
        visualize_detection(model, image_path=image_path, selected_classes=selected_classes,
                            score_thr=score_thr, use_patch_inference=use_patch_inference)
"""


# ===========================================================================
# FAIR1M notebook
# ===========================================================================

def build_fair() -> None:
    cells = []
    cells.append(md("""
# FAIR1M + Strip R-CNN-S Notebook

This notebook is tailored to the FAIR1M-trained Strip R-CNN-S checkpoint
(`weights/strip_rcnn_s_fair1m.pth`). It provides three workflows:

- load the pretrained Strip R-CNN-S detector
- render detections for only the classes you choose
- **evaluate the labelled split with mAP**

**Why we evaluate the validation/train split, not test.** FAIR1M's official
`test` split ships **without public labels** — it is scored only by the ISPRS
online server. So, exactly as for DOTA, the only honest local metric comes from
a split that has ground-truth labels. The base config (`fairv1.py`) exposes the
labelled `train/` split as its `val` dataset, and that is what this notebook
evaluates by default. Point `FAIR_ROOT` at your `fair1_0_to_DOTA_split` layout
(`train/images`, `train/annfiles`, `test/images`).
"""))

    cells.append(md("""
## FAIR1M Class Note

FAIR1M uses 37 fine-grained categories (aircraft types such as `Boeing737`,
ships such as `Passenger_Ship`/`Warship`, vehicles such as `Small_Car`/`Bus`,
plus a few fields/courts). There is no literal `car`/`train` label. Always
confirm `model.CLASSES` after loading; set `RENDER_CLASSES` to the fine-grained
names you care about.
"""))

    cells.append(code(IMPORTS_CELL))

    cells.append(code(f"""
FAIR_CLASSES = (
    'Boeing737', 'Boeing777', 'Boeing747', 'Boeing787', 'A321', 'A220', 'A330',
    'A350', 'C919', 'ARJ21', 'other-airplane', 'Passenger_Ship', 'Motorboat',
    'Fishing_Boat', 'Tugboat', 'Engineering_Ship', 'Liquid_Cargo_Ship',
    'Dry_Cargo_Ship', 'Warship', 'other-ship', 'Small_Car', 'Bus', 'Cargo_Truck',
    'Dump_Truck', 'Van', 'Trailer', 'Tractor', 'Truck_Tractor', 'Excavator',
    'other-vehicle', 'Baseball_Field', 'Basketball_Court', 'Football_Field',
    'Tennis_Court', 'Roundabout', 'Intersection', 'Bridge',
)

CONFIG_PATH = REPO_ROOT / 'configs/strip_rcnn/strip_rcnn_s_fpn_1x_fair_le90.py'
CHECKPOINT_PATH = REPO_ROOT / 'weights/strip_rcnn_s_fair1m.pth'
# FAIR1M split converted to DOTA format. Edit to match where your data lives.
FAIR_ROOT = REPO_ROOT / 'mmrotate/data/fair1_0_to_DOTA_split'
TEST_IMAGE_DIR = FAIR_ROOT / 'test/images'

# FAIR1M test split has no public labels (online-server scored). The labelled
# split used for local mAP is the train split (this mirrors fairv1.py's val).
TEST_SPLIT_HAS_LABELS = False
{DEVICE_BLOCK}
RENDER_CLASSES = ['Passenger_Ship', 'Warship', 'Small_Car', 'Boeing737']
VIS_SCORE_THR = 0.30
USE_PATCH_INFERENCE = False
PATCH_SIZES = [1024]
PATCH_STEPS = [824]
IMG_RATIOS = [1.0]
MERGE_IOU_THR = 0.10

# --- Evaluation defaults: validation split WITH labels -> mAP ---
EVAL_SPLIT = 'val'      # 'val' == labelled train split for FAIR1M
METRIC = 'mAP'
SAMPLES_PER_GPU = 1
WORKERS_PER_GPU = 2
SELECTED_IMAGE_IDS = []
# CPU eval is slow (~tens of seconds/image). Start with a subset; set to None
# for the full split once you confirm it works.
MAX_SUBSET_IMAGES = 20 if DEVICE == 'cpu' else None

MAX_VIS_IMAGES = 8
SAMPLE_IMAGE = None
TEST_IMAGE_CANDIDATES = sorted(TEST_IMAGE_DIR.glob('*.png')) + sorted(TEST_IMAGE_DIR.glob('*.tif'))
if TEST_IMAGE_CANDIDATES:
    SAMPLE_IMAGE = TEST_IMAGE_CANDIDATES[0]

print('Config path:', CONFIG_PATH)
print('Checkpoint path:', CHECKPOINT_PATH)
print('FAIR root:', FAIR_ROOT)
print('CUDA device:', CUDA_DEVICE_NAME, '| capability:', CUDA_CAPABILITY)
print('Supported torch CUDA arches:', SUPPORTED_CUDA_ARCHES)
print('Device:', DEVICE)
print('Eval split / metric:', EVAL_SPLIT, '/', METRIC)
print('Default subset size:', MAX_SUBSET_IMAGES)
print('Test images found:', len(TEST_IMAGE_CANDIDATES))
"""))

    cells.append(code(f"""
def validate_render_classes(selected_classes, all_classes):
    selected = list(selected_classes or all_classes)
    unknown = [class_name for class_name in selected if class_name not in all_classes]
    if unknown:
        raise ValueError(f'Unknown class names: {{unknown}}. Valid FAIR1M classes are: {{all_classes}}')
    return selected


def configure_fair_paths(cfg: Config, dataset_root: Path) -> Config:
    cfg = copy.deepcopy(cfg)
    dataset_root = Path(dataset_root)

    if cfg.model.get('backbone') and cfg.model.backbone.get('init_cfg'):
        cfg.model.backbone.init_cfg = None
    if cfg.model.get('pretrained', None) is not None:
        cfg.model.pretrained = None

    # FAIR1M-in-DOTA-format layout (see configs/_base_/datasets/fairv1.py).
    cfg.data.train.ann_file = str(dataset_root / 'train/annfiles/')
    cfg.data.train.img_prefix = str(dataset_root / 'train/images/')
    # The labelled train split doubles as the val set for local mAP.
    cfg.data.val.ann_file = str(dataset_root / 'train/annfiles/')
    cfg.data.val.img_prefix = str(dataset_root / 'train/images/')
    cfg.data.test.ann_file = str(dataset_root / 'test/images/')
    cfg.data.test.img_prefix = str(dataset_root / 'test/images/')
    return cfg


def build_fair_eval_dataset_cfg(cfg: Config, dataset_root: Path, split: str = 'val'):
    dataset_root = Path(dataset_root)
    split = split.lower()
    if split in ('val', 'train'):
        dataset_cfg = copy.deepcopy(cfg.data.val)
        dataset_cfg['ann_file'] = str(dataset_root / 'train/annfiles/')
        dataset_cfg['img_prefix'] = str(dataset_root / 'train/images/')
    elif split == 'test':
        dataset_cfg = copy.deepcopy(cfg.data.test)
        dataset_cfg['ann_file'] = str(dataset_root / 'test/images/')
        dataset_cfg['img_prefix'] = str(dataset_root / 'test/images/')
    else:
        raise ValueError("split must be one of: 'val', 'train', 'test'")
    dataset_cfg['test_mode'] = True
    return dataset_cfg


def load_strip_rcnn_model(config_path, checkpoint_path, dataset_root, device=DEVICE):
    config_path, checkpoint_path, dataset_root = Path(config_path), Path(checkpoint_path), Path(dataset_root)
    if not config_path.exists():
        raise FileNotFoundError(f'Config file not found: {{config_path}}')
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f'Checkpoint not found: {{checkpoint_path}}. Download the FAIR1M Strip R-CNN-S detector checkpoint.'
        )
    cfg = Config.fromfile(str(config_path))
    cfg = configure_fair_paths(cfg, dataset_root)
    model = init_detector(cfg, str(checkpoint_path), device=device)
    if device == 'cpu':
        model = revert_sync_batchnorm(model)
        model.cfg = cfg
    return model, cfg

{HELPERS_COMMON}
"""))

    cells.append(code("""
model, cfg = load_strip_rcnn_model(CONFIG_PATH, CHECKPOINT_PATH, FAIR_ROOT, device=DEVICE)
print('Checkpoint classes:', model.CLASSES)
"""))

    cells.append(md("## Qualitative inference (single image)"))
    cells.append(code("""
if SAMPLE_IMAGE is None:
    print('No test images found under', TEST_IMAGE_DIR, '- skip this cell or point FAIR_ROOT at your data.')
else:
    raw_result, filtered_result = visualize_detection(
        model, image_path=SAMPLE_IMAGE, selected_classes=RENDER_CLASSES,
        score_thr=VIS_SCORE_THR, use_patch_inference=USE_PATCH_INFERENCE,
    )
"""))

    cells.append(md("## Qualitative inference (several images)"))
    cells.append(code("""
candidate_images = TEST_IMAGE_CANDIDATES[:MAX_VIS_IMAGES]
print([image_path.name for image_path in candidate_images])
visualize_image_list(model, image_paths=candidate_images, selected_classes=RENDER_CLASSES,
                     score_thr=VIS_SCORE_THR, use_patch_inference=USE_PATCH_INFERENCE)
"""))

    cells.append(md("""
## Evaluation (validation split with labels -> mAP)

`MAX_SUBSET_IMAGES` keeps the CPU run responsive. Run the subset cell first;
only then run the full split (it can take a long time on CPU).
"""))
    cells.append(code(eval_helpers_cell('fair', 'dir')))

    cells.append(md("### Subset mAP (quick check)"))
    cells.append(code("""
subset_dataset_obj, subset_outputs, subset_metrics = evaluate_fair_split(
    model=model, cfg=cfg, dataset_root=FAIR_ROOT, split=EVAL_SPLIT, metric=METRIC,
    selected_image_ids=SELECTED_IMAGE_IDS, max_subset_images=MAX_SUBSET_IMAGES,
    samples_per_gpu=SAMPLES_PER_GPU, workers_per_gpu=WORKERS_PER_GPU,
)
print(f'Images evaluated in subset: {len(subset_dataset_obj)}')
subset_metrics
"""))

    cells.append(md("### Full-split mAP (slow on CPU — run when ready)"))
    cells.append(code("""
# Set max_subset_images=None to evaluate the whole labelled split.
full_dataset, full_outputs, full_metrics = evaluate_fair_split(
    model=model, cfg=cfg, dataset_root=FAIR_ROOT, split=EVAL_SPLIT, metric=METRIC,
    selected_image_ids=None, max_subset_images=None,
    samples_per_gpu=SAMPLES_PER_GPU, workers_per_gpu=WORKERS_PER_GPU,
)
print(f'Images evaluated: {len(full_dataset)}')
full_metrics
"""))

    cells.append(md("""
## Recommended usage

1. Confirm `CHECKPOINT_PATH` and `FAIR_ROOT` point to your local files.
2. Run the setup cells; `DEVICE` will fall back to CPU on the RTX 5070 Ti
   (sm_120) because this torch 1.8 build only supports up to sm_75.
3. Use the single/multi-image cells for qualitative checks on `test/images`.
4. Run the **subset mAP** cell first; increase `MAX_SUBSET_IMAGES` or run the
   full-split cell when ready.
5. The FAIR1M `test` split has no public labels — for an official number,
   format predictions and submit to the ISPRS server.
"""))

    write_nb(cells, NB_DIR / 'fair1m_strip_rcnn_s_workflow.ipynb')


# ===========================================================================
# HRSC2016 notebook
# ===========================================================================

def build_hrsc() -> None:
    cells = []
    cells.append(md("""
# HRSC2016 + Strip R-CNN-S Notebook

This notebook is tailored to an HRSC2016-trained Strip R-CNN-S checkpoint. It
provides three workflows:

- load the pretrained Strip R-CNN-S detector
- render ship detections
- **evaluate the labelled HRSC test split with mAP**

Unlike DOTA and FAIR1M, the HRSC2016 `test` split **does** ship with
ground-truth annotations locally, so we evaluate it directly (this is still the
"validation-with-labels" idea — we score against a split whose labels we hold).
"""))

    cells.append(md("""
## Two things to set up first

1. **Checkpoint.** This repo ships the DOTA and FAIR1M detector weights but
   **not** an HRSC one. HRSC is single-class (`ship`, `num_classes=1`), so a
   DOTA/FAIR checkpoint will *not* load into it. Download or train an HRSC
   Strip R-CNN-S checkpoint and point `CHECKPOINT_PATH` at it.
2. **Data layout.** The config expects, under `HRSC_ROOT`:
   `ImageSets/{trainval,test}.txt`, `FullDataSet/AllImages/<id>.bmp`,
   `FullDataSet/Annotations/<id>.xml`. The `archive/` download in
   `mmrotate/data/hrsc` still needs to be extracted into that layout.
"""))

    cells.append(code(IMPORTS_CELL))

    cells.append(code(f"""
# HRSC is single-class by default (classwise=False -> just 'ship').
HRSC_CLASSES = ('ship',)

CONFIG_PATH = REPO_ROOT / 'configs/strip_rcnn/strip_rcnn_s_fpn_3x_hrsc_le90.py'
# No HRSC checkpoint ships with this repo - update this path.
CHECKPOINT_PATH = REPO_ROOT / 'weights/strip_rcnn_s_hrsc.pth'
HRSC_ROOT = REPO_ROOT / 'mmrotate/data/HRSC2016'

IMAGE_DIR = HRSC_ROOT / 'FullDataSet/AllImages'
ANN_SUBDIR = HRSC_ROOT / 'FullDataSet/Annotations'
TRAINVAL_SET = HRSC_ROOT / 'ImageSets/trainval.txt'
TEST_SET = HRSC_ROOT / 'ImageSets/test.txt'

# The HRSC test split is labelled locally, so mAP on it is valid.
TEST_SPLIT_HAS_LABELS = True
{DEVICE_BLOCK}
RENDER_CLASSES = ['ship']
VIS_SCORE_THR = 0.30
USE_PATCH_INFERENCE = False
PATCH_SIZES = [800]
PATCH_STEPS = [650]
IMG_RATIOS = [1.0]
MERGE_IOU_THR = 0.10

# --- Evaluation defaults: test split WITH labels -> mAP ---
EVAL_SPLIT = 'test'     # HRSC test.txt is labelled
METRIC = 'mAP'
SAMPLES_PER_GPU = 1
WORKERS_PER_GPU = 2
SELECTED_IMAGE_IDS = []
MAX_SUBSET_IMAGES = 20 if DEVICE == 'cpu' else None

MAX_VIS_IMAGES = 8
SAMPLE_IMAGE = None
TEST_IMAGE_CANDIDATES = sorted(IMAGE_DIR.glob('*.bmp')) if IMAGE_DIR.exists() else []
if TEST_IMAGE_CANDIDATES:
    SAMPLE_IMAGE = TEST_IMAGE_CANDIDATES[0]

print('Config path:', CONFIG_PATH)
print('Checkpoint path:', CHECKPOINT_PATH)
print('HRSC root:', HRSC_ROOT)
print('Image dir exists:', IMAGE_DIR.exists(), '| Test set exists:', TEST_SET.exists())
print('CUDA device:', CUDA_DEVICE_NAME, '| capability:', CUDA_CAPABILITY)
print('Supported torch CUDA arches:', SUPPORTED_CUDA_ARCHES)
print('Device:', DEVICE)
print('Eval split / metric:', EVAL_SPLIT, '/', METRIC)
print('Default subset size:', MAX_SUBSET_IMAGES)
print('Images found:', len(TEST_IMAGE_CANDIDATES))
"""))

    cells.append(code(f"""
def validate_render_classes(selected_classes, all_classes):
    selected = list(selected_classes or all_classes)
    unknown = [class_name for class_name in selected if class_name not in all_classes]
    if unknown:
        raise ValueError(f'Unknown class names: {{unknown}}. Valid HRSC classes are: {{all_classes}}')
    return selected


def configure_hrsc_paths(cfg: Config, dataset_root: Path) -> Config:
    cfg = copy.deepcopy(cfg)
    dataset_root = Path(dataset_root)

    if cfg.model.get('backbone') and cfg.model.backbone.get('init_cfg'):
        cfg.model.backbone.init_cfg = None
    if cfg.model.get('pretrained', None) is not None:
        cfg.model.pretrained = None

    # Absolute paths matter here: HRSCDataset builds the xml path as
    # osp.join(img_prefix, ann_subdir, '<id>.xml'), which only resolves
    # correctly when ann_subdir is absolute.
    image_dir = str(dataset_root / 'FullDataSet/AllImages') + '/'
    ann_subdir = str(dataset_root / 'FullDataSet/Annotations') + '/'
    for split, imageset in (('train', 'trainval.txt'), ('val', 'test.txt'), ('test', 'test.txt')):
        node = cfg.data[split]
        node.ann_file = str(dataset_root / 'ImageSets' / imageset)
        node.img_prefix = image_dir
        node.img_subdir = image_dir
        node.ann_subdir = ann_subdir
    return cfg


def build_hrsc_eval_dataset_cfg(cfg: Config, dataset_root: Path, split: str = 'test'):
    dataset_root = Path(dataset_root)
    split = split.lower()
    imageset = {{'train': 'trainval.txt', 'trainval': 'trainval.txt',
                'val': 'test.txt', 'test': 'test.txt'}}.get(split)
    if imageset is None:
        raise ValueError("split must be one of: 'train', 'val', 'test'")
    dataset_cfg = copy.deepcopy(cfg.data.test)
    dataset_cfg['ann_file'] = str(dataset_root / 'ImageSets' / imageset)
    dataset_cfg['img_prefix'] = str(dataset_root / 'FullDataSet/AllImages') + '/'
    dataset_cfg['img_subdir'] = dataset_cfg['img_prefix']
    dataset_cfg['ann_subdir'] = str(dataset_root / 'FullDataSet/Annotations') + '/'
    dataset_cfg['test_mode'] = True
    return dataset_cfg


def load_strip_rcnn_model(config_path, checkpoint_path, dataset_root, device=DEVICE):
    config_path, checkpoint_path, dataset_root = Path(config_path), Path(checkpoint_path), Path(dataset_root)
    if not config_path.exists():
        raise FileNotFoundError(f'Config file not found: {{config_path}}')
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f'Checkpoint not found: {{checkpoint_path}}. This repo does not ship an HRSC '
            f'Strip R-CNN-S detector; download or train one and update CHECKPOINT_PATH.'
        )
    cfg = Config.fromfile(str(config_path))
    cfg = configure_hrsc_paths(cfg, dataset_root)
    model = init_detector(cfg, str(checkpoint_path), device=device)
    if device == 'cpu':
        model = revert_sync_batchnorm(model)
        model.cfg = cfg
    return model, cfg

{HELPERS_COMMON}
"""))

    cells.append(code("""
model, cfg = load_strip_rcnn_model(CONFIG_PATH, CHECKPOINT_PATH, HRSC_ROOT, device=DEVICE)
print('Checkpoint classes:', model.CLASSES)
"""))

    cells.append(md("## Qualitative inference (single image)"))
    cells.append(code("""
if SAMPLE_IMAGE is None:
    print('No .bmp images found under', IMAGE_DIR, '- extract HRSC2016 into the expected layout first.')
else:
    raw_result, filtered_result = visualize_detection(
        model, image_path=SAMPLE_IMAGE, selected_classes=RENDER_CLASSES,
        score_thr=VIS_SCORE_THR, use_patch_inference=USE_PATCH_INFERENCE,
    )
"""))

    cells.append(md("## Qualitative inference (several images)"))
    cells.append(code("""
candidate_images = TEST_IMAGE_CANDIDATES[:MAX_VIS_IMAGES]
print([image_path.name for image_path in candidate_images])
visualize_image_list(model, image_paths=candidate_images, selected_classes=RENDER_CLASSES,
                     score_thr=VIS_SCORE_THR, use_patch_inference=USE_PATCH_INFERENCE)
"""))

    cells.append(md("""
## Evaluation (HRSC test split with labels -> mAP)

HRSC evaluation reports VOC-style AP (the dataset class uses the 2007 metric by
default). Run the subset cell first, then the full split.
"""))
    cells.append(code(eval_helpers_cell('hrsc', 'imageset')))

    cells.append(md("### Subset mAP (quick check)"))
    cells.append(code("""
subset_dataset_obj, subset_outputs, subset_metrics = evaluate_hrsc_split(
    model=model, cfg=cfg, dataset_root=HRSC_ROOT, split=EVAL_SPLIT, metric=METRIC,
    selected_image_ids=SELECTED_IMAGE_IDS, max_subset_images=MAX_SUBSET_IMAGES,
    samples_per_gpu=SAMPLES_PER_GPU, workers_per_gpu=WORKERS_PER_GPU,
)
print(f'Images evaluated in subset: {len(subset_dataset_obj)}')
subset_metrics
"""))

    cells.append(md("### Full-split mAP (slow on CPU — run when ready)"))
    cells.append(code("""
full_dataset, full_outputs, full_metrics = evaluate_hrsc_split(
    model=model, cfg=cfg, dataset_root=HRSC_ROOT, split=EVAL_SPLIT, metric=METRIC,
    selected_image_ids=None, max_subset_images=None,
    samples_per_gpu=SAMPLES_PER_GPU, workers_per_gpu=WORKERS_PER_GPU,
)
print(f'Images evaluated: {len(full_dataset)}')
full_metrics
"""))

    cells.append(md("""
## Recommended usage

1. Provide an HRSC Strip R-CNN-S checkpoint and set `CHECKPOINT_PATH`.
2. Extract HRSC2016 into `ImageSets/`, `FullDataSet/AllImages/`,
   `FullDataSet/Annotations/` under `HRSC_ROOT`.
3. Run the setup cells; `DEVICE` falls back to CPU on the RTX 5070 Ti (sm_120).
4. Run the **subset mAP** cell first, then the full-split cell.
"""))

    write_nb(cells, NB_DIR / 'hrsc2016_strip_rcnn_s_workflow.ipynb')


# ===========================================================================
# Patch the existing DOTA notebook -> validation-with-labels default
# ===========================================================================

def patch_dota() -> None:
    path = NB_DIR / 'dota_strip_rcnn_s_workflow.ipynb'
    nb = json.loads(path.read_text())

    def join(cell):
        return ''.join(cell['source'])

    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = join(cell)
        if "EVAL_SPLIT = 'test'" in src and "METRIC = None" in src:
            src = src.replace("EVAL_SPLIT = 'test'", "EVAL_SPLIT = 'val'")
            src = src.replace("METRIC = None", "METRIC = 'mAP'")
            # Make the CPU subset a touch larger now that we actually score.
            src = src.replace(
                "MAX_SUBSET_IMAGES = 8 if DEVICE == 'cpu' else None",
                "MAX_SUBSET_IMAGES = 20 if DEVICE == 'cpu' else None",
            )
            cell['source'] = src.splitlines(keepends=True)
        if 'def evaluate_dota_split' in src and 'TEST_SPLIT_HAS_LABELS' not in src:
            # Keep behaviour identical; just note the convention symbolically.
            pass

    # Insert a TEST_SPLIT_HAS_LABELS flag into the config cell for parity.
    for cell in nb['cells']:
        src = join(cell)
        if cell['cell_type'] == 'code' and 'DOTA_ROOT = REPO_ROOT' in src and 'TEST_SPLIT_HAS_LABELS' not in src:
            src = src.replace(
                "TEST_IMAGE_DIR = DOTA_ROOT / 'test/images'",
                "TEST_IMAGE_DIR = DOTA_ROOT / 'test/images'\n# DOTA test split has no local labels (online-server scored).\nTEST_SPLIT_HAS_LABELS = False",
            )
            cell['source'] = src.splitlines(keepends=True)

    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print("patched", path)


if __name__ == '__main__':
    build_fair()
    build_hrsc()
    patch_dota()
