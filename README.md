# NV Detections Classes Replacement

NovaVision component for replacing generic detection class labels with refined classification results.

This package implements a deterministic post-processing step for two-stage computer vision workflows:

```text
image
  -> object detection
  -> crop detected objects
  -> classification on each crop
  -> detections classes replacement
  -> visualization / filtering / analytics
```

For example, a detector may output a generic `vehicle` detection with a bounding box. A classifier running on the crop can then refine that label to `truck`, `car`, or a brand/model. This component keeps the detection geometry and replaces the class-related fields.

## What It Does

The component receives:

- object detection predictions with bounding boxes and generic class labels
- classification predictions from cropped detections

It returns:

- updated detection predictions with replaced `class`, `class_id`, and `confidence`
- original bounding box fields preserved
- original metadata preserved
- `metadata.original_detection_id` added
- new unique `detection_id` values generated

Unmatched detections are discarded by default. If fallback config is provided, unmatched detections are kept with the fallback class.

Reference behavior is based on Roboflow Inference:

```text
https://inference.roboflow.com/workflows/blocks/detections_classes_replacement/
```

## Package Structure

```text
apps/
  sample_request.json
  run_sample_request.py
notebooks/
  no_train_demo.ipynb
resources/
  benchmark_research_report.md
src/
  executors/
    detections_classes_replacement_executor.py
  models/
    package_model.py
    PackageModel.py
  utils/
tests/
  test_executor.py
  test_package_model.py
Dockerfile.dev
Dockerfile.prod
requirements.dev.txt
requirements.prod.txt
service.py
```

## Inputs

### `inputDetections`

List of detection dictionaries.

Expected detection fields:

```json
{
  "detection_id": "det-vehicle-1",
  "x": 120.0,
  "y": 80.0,
  "width": 64.0,
  "height": 40.0,
  "confidence": 0.82,
  "class": "vehicle",
  "class_id": 1,
  "metadata": {
    "source": "generic-detector"
  }
}
```

### `inputData`

List of classification results from the classifier stage. It uses the standard NovaVision `inputData` socket name so it can be connected on the flow canvas. The executor supports:

- dictionaries linked with `parent_id`
- single-label classification dictionaries with `top`
- multi-label classification dictionaries with `predictions`
- plain strings
- lists of strings

Examples:

```json
{
  "parent_id": "det-vehicle-1",
  "top": "truck",
  "class_id": 7,
  "confidence": 0.94
}
```

```json
{
  "parent_id": "det-animal-1",
  "predictions": [
    {
      "class": "cat",
      "class_id": 12,
      "confidence": 0.21
    },
    {
      "class": "golden_retriever",
      "class_id": 18,
      "confidence": 0.89
    }
  ]
}
```

## Configs

### `MatchingMode`

Default: `Auto`

Options:

- `Auto`: first match by `parent_id` / `detection_id`, then use positional matching for raw strings, string lists, or dictionaries without parent IDs.
- `ParentIdOnly`: only match classification outputs by parent/detection ID.
- `PositionalOnly`: match detection and classification outputs by list order.

### `FallbackClassName`

Default: empty string.

If empty, unmatched detections are discarded. If set, unmatched detections are kept and assigned this class name.

### `FallbackClassId`

Default: `-1`.

Only used when `FallbackClassName` is set. If the value is negative, the executor resolves it to `sys.maxsize`.

## Output

### `outputDetections`

List of updated detection dictionaries.

Example output:

```json
{
  "detection_id": "new-generated-uuid",
  "x": 120.0,
  "y": 80.0,
  "width": 64.0,
  "height": 40.0,
  "confidence": 0.94,
  "class": "truck",
  "class_id": 7,
  "metadata": {
    "source": "generic-detector",
    "original_detection_id": "det-vehicle-1"
  }
}
```

## Matching And Replacement Rules

- Classification dictionaries with `parent_id`, `parentId`, `detection_id`, or `original_detection_id` can match a detection by `detection_id`.
- String predictions are matched positionally and become the new class name with `class_id = 0` and `confidence = 1.0`.
- String-list predictions are matched positionally and use the first non-empty string.
- Single-label classification uses `top`.
- Multi-label classification uses the item with the highest confidence in `predictions`.
- Original bounding box fields are kept unchanged.
- Original metadata is kept and `metadata.original_detection_id` is added.
- New detection IDs are generated with UUIDs.

## Install

Production dependencies:

```powershell
python -m pip install -r requirements.prod.txt
```

Development dependencies:

```powershell
python -m pip install -r requirements.dev.txt
```

Clean virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.dev.txt
```

## Run Tests

```powershell
python -m pytest
```

Expected result:

```text
15 passed
```

With clean virtual environment:

```powershell
.venv\Scripts\python -m pytest
```

## Run Service Bootstrap

```powershell
python service.py
```

Expected result:

```text
DetectionsClassesReplacementExecutor: ready
```

With clean virtual environment:

```powershell
.venv\Scripts\python service.py
```

## Run Client App

Run the local sample request:

```powershell
python apps/run_sample_request.py
```

Expected summary:

```text
det-vehicle-1: vehicle -> truck (confidence 0.94)
det-animal-1: animal -> golden_retriever (confidence 0.89)
```

If a compatible NovaVision service is already running and exposes `/api`, the same client can post the sample payload by setting:

```powershell
$env:NOVAVISION_API_URL = "http://127.0.0.1:8000"
python apps/run_sample_request.py
```

## Docker

Build development image:

```powershell
docker build -f Dockerfile.dev -t nv-dcr:dev .
```

Build production image:

```powershell
docker build -f Dockerfile.prod -t nv-dcr:prod .
```

Run image bootstrap:

```powershell
docker run --rm nv-dcr:dev
docker run --rm nv-dcr:prod
```

Expected result:

```text
DetectionsClassesReplacementExecutor: ready
```

## No Training Or Dataset Required

This package does not train a model. It does not include model weights and does not need a dataset.

The detector, cropper, and classifier are upstream workflow steps. This component only combines their already-produced outputs and updates detection classes. The notebook in `notebooks/no_train_demo.ipynb` demonstrates this with synthetic sample data.

## Branch Workflow

The repository follows the expected branch shape:

```text
main
develop
feature/detections-classes-replacement
```

Current work should continue on:

```text
feature/detections-classes-replacement
```

The normal merge direction is:

```text
feature branch -> develop -> main
```

## Checklist Mapping

| Checklist item | Project artifact / validation |
| --- | --- |
| Package Init | Base package structure, `service.py`, requirements, `src/`, `apps/`, `tests/`, `resources/`, `notebooks/` |
| Libraries / Tech / Dataset (Benchmark Research Report) | `resources/benchmark_research_report.md` |
| Image Build | `Dockerfile.dev`, `Dockerfile.prod`, successful Docker builds |
| Model Design (PackageModel) | `src/models/package_model.py`, `src/models/PackageModel.py`, package model tests |
| Model Train Notebook (Colab/Local) | `notebooks/no_train_demo.ipynb` |
| Executor Implement (`{{name}}.py`) | `src/executors/detections_classes_replacement_executor.py`, executor tests |
| Client App | `apps/sample_request.json`, `apps/run_sample_request.py` |
| Clean Install Test (Dev/Prod) | Clean `.venv` install, pytest, service bootstrap, dev/prod Docker builds |
| Developer and User Guide (`README.md`) | This file |

## Notes

Local Pydantic v2 may print warnings about v1-style `class Config` keys. The package keeps this metadata intentionally because NovaVision package model examples and prior Suite compatibility work use Pydantic `class Config` metadata for titles and schema targets.

NovaVision canvas socket names are intentionally lower-camel-case:

- `inputDetections`
- `inputData`
- `outputDetections`

Using names such as `ObjectDetectionPredictions` or custom names such as `inputClassificationPredictions` in the visible package model can prevent the Suite flow builder from rendering input/output ports.

