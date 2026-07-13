# Benchmark Research Report: Detections Classes Replacement

## Purpose

The Detections Classes Replacement component is a post-processing block for two-stage computer vision workflows. A detector first finds object locations with generic labels, then a classifier runs on each detected crop and provides more specific labels. This component combines both outputs by replacing each detection's generic class information with the matching classifier result.

Typical workflow:

```text
image -> object detection -> crop detections -> classification -> detections classes replacement -> visualization/filtering/analytics
```

This is useful when one model is strong at localization and another model is strong at fine-grained recognition. Example: a detector outputs `vehicle`, then a classifier replaces it with `car`, `truck`, or a brand/model label.

## Reference Behavior

Primary reference:

- Roboflow Inference, Detections Classes Replacement: https://inference.roboflow.com/workflows/blocks/detections_classes_replacement/
- Roboflow Inference source package: https://github.com/roboflow/inference

Reference class and identifier:

- Class: `DetectionsClassesReplacementBlockV1`
- Roboflow workflow type: `roboflow_core/detections_classes_replacement@v1`

The reference block:

- Receives object detection predictions containing bounding boxes and generic class labels.
- Receives classification predictions produced from crops of those detections.
- Matches classifications to detections by `parent_id` / detection ID when available.
- Uses positional matching for raw string or list labels.
- For single-label classification, uses the `top` class.
- For multi-label classification, uses the class with the highest confidence.
- For string labels, uses the string as the new class name with default confidence `1.0` and class ID `0`.
- Replaces class name, class ID, and confidence on the detection.
- Preserves bounding box coordinates and other detection geometry.
- Generates new detection IDs for updated detections.
- Discards unmatched detections unless fallback configuration is provided.

## Inputs And Outputs

Required inputs:

```text
object_detection_predictions
classification_predictions
```

`object_detection_predictions` should contain detection records with bounding box data and class data. For the NovaVision MVP, the package will use serialized dictionary/list structures instead of requiring Roboflow's internal `sv.Detections` object.

Expected detection fields:

```text
detection_id
x
y
width
height
confidence
class
class_id
metadata
```

`classification_predictions` may be one of:

- classification dictionaries linked to a detection with `parent_id`
- single-label classification dictionaries with `top`
- multi-label classification dictionaries with class confidence values
- plain strings, such as OCR or VLM labels
- lists of strings, where the first usable label can be used positionally

Optional configs:

```text
fallback_class_name
fallback_class_id
matching_mode
```

Output:

```text
predictions
```

The output should be the filtered and updated detection list. It should keep detection geometry, replace class fields, and include new detection IDs.

## Libraries And Technology

Recommended package-level technology:

- Python 3.10+
- Pydantic `>=1,<3` for NovaVision-compatible request/response models
- Python standard library for replacement logic: `copy`, `sys`, `uuid`, and typing helpers
- Pytest for validation tests

The Roboflow implementation uses Roboflow Inference workflow abstractions, `supervision`, and NumPy internally. For this NovaVision component, those heavy dependencies are not required for the first implementation because the component can operate on serialized prediction dictionaries. This keeps the package easier to test locally and avoids unnecessary model/runtime dependencies.

## Dataset And Training Decision

No dataset is required for this component.

Reason: the component does not train a detector or classifier. It only transforms model outputs that already exist. The detector, cropper, and classifier are upstream workflow blocks. This package receives their predictions and produces updated detections.

No model training is required.

The notebook checklist item should be satisfied with a no-training demo notebook that explains the decision and demonstrates sample detection/classification replacement with synthetic prediction data.

## Edge Cases To Test

The implementation should cover these cases:

- Classification `parent_id` matches a detection's `detection_id`.
- Classification order matches detection order when using positional labels.
- A single-label classifier result has a `top` class.
- A multi-label classifier result has multiple classes and one highest confidence.
- A string prediction is used as the class name.
- A list of strings is reduced to a usable class label.
- Detection has no matching classification and no fallback, so it is removed.
- Detection has no matching classification but fallback is set, so fallback class is used.
- Fallback class ID is missing or negative, so it resolves to `sys.maxsize`.
- Bounding box fields are preserved after class replacement.
- Updated detections receive new unique `detection_id` values.
- Empty detection input returns an empty output.
- Empty classification input returns an empty output unless fallback behavior is explicitly applied.

## Benchmark / Acceptance Criteria

Because this is deterministic post-processing logic, the benchmark is correctness-based instead of accuracy-based.

Minimum acceptance criteria:

- Unit tests pass for all matching, fallback, and class extraction scenarios.
- Output detection count is correct after filtering or fallback.
- Output classes, class IDs, and confidences match classifier or fallback data.
- Original bounding boxes are unchanged.
- New detection IDs are unique and different from original detection IDs.
- The component can run without a model file, dataset, GPU, or external API key.

## Checklist Result

This report satisfies:

```text
Libraries / Tech / Dataset (Benchmark Research Report)
```

