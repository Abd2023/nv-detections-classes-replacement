import sys

from src.executors import DetectionsClassesReplacementExecutor
from src.models import (
    DetectionsClassesReplacementRequest,
    FallbackClassName,
    ParentIdOnlyMatchingModeOption,
    PositionalOnlyMatchingModeOption,
)
from src.models.package_model import FallbackClassId, MatchingMode


def run_executor(*, detections, classifications, configs=None):
    request = DetectionsClassesReplacementRequest(
        inputs={
            "inputDetections": {"value": detections},
            "inputData": {"value": classifications},
        },
        configs=configs or {},
    )
    return DetectionsClassesReplacementExecutor(request).run().outputs.predictions.value


def test_parent_id_replacement_uses_single_label_top():
    outputs = run_executor(
        detections=[
            {
                "detection_id": "det-1",
                "x": 100,
                "y": 80,
                "width": 40,
                "height": 30,
                "confidence": 0.82,
                "class": "vehicle",
                "class_id": 1,
                "metadata": {"source": "detector"},
            }
        ],
        classifications=[
            {
                "parent_id": "det-1",
                "top": "truck",
                "class_id": 7,
                "confidence": 0.94,
            }
        ],
    )

    assert len(outputs) == 1
    assert outputs[0].class_name == "truck"
    assert outputs[0].class_id == 7
    assert outputs[0].confidence == 0.94


def test_positional_string_replacement_defaults_class_id_and_confidence():
    outputs = run_executor(
        detections=[
            {
                "detection_id": "det-1",
                "x": 1,
                "y": 2,
                "width": 3,
                "height": 4,
                "confidence": 0.5,
                "class": "text",
                "class_id": 2,
                "metadata": {},
            }
        ],
        classifications=["invoice_number"],
    )

    assert len(outputs) == 1
    assert outputs[0].class_name == "invoice_number"
    assert outputs[0].class_id == 0
    assert outputs[0].confidence == 1.0


def test_positional_only_matches_string_lists():
    outputs = run_executor(
        detections=[
            {
                "detection_id": "det-1",
                "x": 1,
                "y": 2,
                "width": 3,
                "height": 4,
                "confidence": 0.5,
                "class": "ocr",
                "class_id": 2,
                "metadata": {},
            }
        ],
        classifications=[["plate_34", "plate_35"]],
        configs={
            "MatchingMode": MatchingMode(value=PositionalOnlyMatchingModeOption()),
        },
    )

    assert len(outputs) == 1
    assert outputs[0].class_name == "plate_34"
    assert outputs[0].class_id == 0
    assert outputs[0].confidence == 1.0


def test_multi_label_classification_uses_highest_confidence():
    outputs = run_executor(
        detections=[
            {
                "detection_id": "det-animal",
                "x": 260,
                "y": 140,
                "width": 72,
                "height": 58,
                "confidence": 0.76,
                "class": "animal",
                "class_id": 2,
                "metadata": {},
            }
        ],
        classifications=[
            {
                "parent_id": "det-animal",
                "predictions": [
                    {"class": "cat", "class_id": 12, "confidence": 0.21},
                    {"class": "golden_retriever", "class_id": 18, "confidence": 0.89},
                ],
            }
        ],
    )

    assert len(outputs) == 1
    assert outputs[0].class_name == "golden_retriever"
    assert outputs[0].class_id == 18
    assert outputs[0].confidence == 0.89


def test_unmatched_detection_is_discarded_without_fallback():
    outputs = run_executor(
        detections=[
            {
                "detection_id": "det-1",
                "x": 1,
                "y": 2,
                "width": 3,
                "height": 4,
                "confidence": 0.5,
                "class": "vehicle",
                "class_id": 1,
                "metadata": {},
            }
        ],
        classifications=[],
    )

    assert outputs == []


def test_fallback_applied_with_sys_maxsize_for_negative_class_id():
    outputs = run_executor(
        detections=[
            {
                "detection_id": "det-1",
                "x": 1,
                "y": 2,
                "width": 3,
                "height": 4,
                "confidence": 0.57,
                "class": "vehicle",
                "class_id": 1,
                "metadata": {},
            }
        ],
        classifications=[],
        configs={
            "FallbackClassName": FallbackClassName(value="unknown"),
            "FallbackClassId": FallbackClassId(value=-1),
        },
    )

    assert len(outputs) == 1
    assert outputs[0].class_name == "unknown"
    assert outputs[0].class_id == sys.maxsize
    assert outputs[0].confidence == 0.57


def test_box_coordinates_metadata_and_new_detection_id_are_preserved_or_updated():
    outputs = run_executor(
        detections=[
            {
                "detection_id": "det-box",
                "x": 10.5,
                "y": 20.5,
                "width": 30.5,
                "height": 40.5,
                "confidence": 0.81,
                "class": "vehicle",
                "class_id": 1,
                "metadata": {"source": "detector"},
            }
        ],
        classifications=[
            {
                "parent_id": "det-box",
                "top": "bus",
                "class_id": 9,
                "confidence": 0.93,
            }
        ],
    )

    assert len(outputs) == 1
    output = outputs[0]
    assert output.detection_id != "det-box"
    assert output.x == 10.5
    assert output.y == 20.5
    assert output.width == 30.5
    assert output.height == 40.5
    assert output.metadata["source"] == "detector"
    assert output.metadata["original_detection_id"] == "det-box"


def test_parent_id_only_does_not_fallback_to_position():
    outputs = run_executor(
        detections=[
            {
                "detection_id": "det-1",
                "x": 1,
                "y": 2,
                "width": 3,
                "height": 4,
                "confidence": 0.5,
                "class": "vehicle",
                "class_id": 1,
                "metadata": {},
            }
        ],
        classifications=["truck"],
        configs={
            "MatchingMode": MatchingMode(value=ParentIdOnlyMatchingModeOption()),
        },
    )

    assert outputs == []
