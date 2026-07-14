from __future__ import annotations

from copy import deepcopy
import sys
from typing import Any
from uuid import uuid4

from ..models import (
    DetectionPrediction,
    DetectionsClassesReplacementOutputs,
    DetectionsClassesReplacementRequest,
    DetectionsClassesReplacementResponse,
    Predictions,
)


class DetectionsClassesReplacementExecutor:
    """Replace generic detection classes with matching classification outputs."""

    def __init__(self, request: Any, bootstrap: dict[str, Any] | None = None) -> None:
        self.request = self._parse_request(request)
        self.bootstrap_data = bootstrap or {}

    @staticmethod
    def bootstrap() -> dict[str, str]:
        return {"executor": "DetectionsClassesReplacementExecutor", "status": "ready"}

    def run(self) -> DetectionsClassesReplacementResponse:
        detections = self.request.inputs.object_detection_predictions.value
        classifications = self.request.inputs.classification_predictions.value
        matching_mode = self.request.configs.matching_mode.value.name
        fallback_class_name = self.request.configs.fallback_class_name.value
        fallback_class_id = self.request.configs.fallback_class_id.value

        classifications_by_parent_id = self._classifications_by_parent_id(classifications)
        updated_detections: list[DetectionPrediction] = []

        for index, detection in enumerate(detections):
            classification = self._find_classification(
                detection=detection,
                index=index,
                classifications=classifications,
                classifications_by_parent_id=classifications_by_parent_id,
                matching_mode=matching_mode,
            )
            replacement = self._extract_replacement(classification) if classification is not None else None

            if replacement is None and fallback_class_name:
                replacement = (
                    fallback_class_name,
                    fallback_class_id if fallback_class_id >= 0 else sys.maxsize,
                    detection.confidence,
                )

            if replacement is None:
                continue

            updated_detections.append(self._replace_detection(detection, replacement))

        return DetectionsClassesReplacementResponse(
            outputs=DetectionsClassesReplacementOutputs(
                predictions=Predictions(value=updated_detections)
            )
        )

    @staticmethod
    def _parse_request(request: Any) -> DetectionsClassesReplacementRequest:
        if isinstance(request, DetectionsClassesReplacementRequest):
            return request
        if hasattr(DetectionsClassesReplacementRequest, "model_validate"):
            return DetectionsClassesReplacementRequest.model_validate(request)
        return DetectionsClassesReplacementRequest.parse_obj(request)

    @staticmethod
    def _classifications_by_parent_id(classifications: list[Any]) -> dict[str, Any]:
        indexed: dict[str, Any] = {}
        for classification in classifications:
            if not isinstance(classification, dict):
                continue
            parent_id = DetectionsClassesReplacementExecutor._get_parent_id(classification)
            if parent_id:
                indexed[str(parent_id)] = classification
        return indexed

    @staticmethod
    def _get_parent_id(classification: dict[str, Any]) -> Any:
        for key in ("parent_id", "parentId", "detection_id", "original_detection_id"):
            if classification.get(key):
                return classification[key]
        return None

    @staticmethod
    def _find_classification(
        *,
        detection: DetectionPrediction,
        index: int,
        classifications: list[Any],
        classifications_by_parent_id: dict[str, Any],
        matching_mode: str,
    ) -> Any | None:
        if matching_mode != "PositionalOnly":
            matched_by_parent = classifications_by_parent_id.get(detection.detection_id)
            if matched_by_parent is not None:
                return matched_by_parent

        if matching_mode == "ParentIdOnly" or index >= len(classifications):
            return None

        positional_candidate = classifications[index]
        if matching_mode == "PositionalOnly":
            return positional_candidate

        if isinstance(positional_candidate, (str, list)):
            return positional_candidate

        if isinstance(positional_candidate, dict) and not DetectionsClassesReplacementExecutor._get_parent_id(
            positional_candidate
        ):
            return positional_candidate

        return None

    @staticmethod
    def _extract_replacement(classification: Any) -> tuple[str, int, float] | None:
        if isinstance(classification, str):
            return (classification, 0, 1.0) if classification else None

        if isinstance(classification, list):
            for label in classification:
                if isinstance(label, str) and label:
                    return label, 0, 1.0
            return None

        if not isinstance(classification, dict):
            return None

        if "top" in classification and classification["top"]:
            top = classification["top"]
            if isinstance(top, dict):
                return DetectionsClassesReplacementExecutor._replacement_from_prediction_dict(top)

            class_id = classification.get("class_id", classification.get("top_class_id", 0))
            confidence = classification.get("confidence", classification.get("top_confidence", 1.0))
            return str(top), int(class_id), float(confidence)

        prediction_items = DetectionsClassesReplacementExecutor._prediction_items(classification)
        if prediction_items:
            best_prediction = max(
                prediction_items,
                key=lambda item: DetectionsClassesReplacementExecutor._prediction_confidence(item),
            )
            return DetectionsClassesReplacementExecutor._replacement_from_prediction_dict(best_prediction)

        direct_class = classification.get("class", classification.get("class_name"))
        if direct_class:
            return (
                str(direct_class),
                int(classification.get("class_id", 0)),
                float(classification.get("confidence", 1.0)),
            )

        return None

    @staticmethod
    def _prediction_items(classification: dict[str, Any]) -> list[dict[str, Any]]:
        predictions = classification.get("predictions", [])
        if isinstance(predictions, list):
            return [item for item in predictions if isinstance(item, dict)]

        if isinstance(predictions, dict):
            items: list[dict[str, Any]] = []
            for class_name, value in predictions.items():
                if isinstance(value, dict):
                    item = dict(value)
                    item.setdefault("class", class_name)
                    items.append(item)
                else:
                    items.append({"class": class_name, "confidence": value})
            return items

        return []

    @staticmethod
    def _prediction_confidence(prediction: dict[str, Any]) -> float:
        return float(prediction.get("confidence", prediction.get("score", prediction.get("probability", 0.0))))

    @staticmethod
    def _replacement_from_prediction_dict(prediction: dict[str, Any]) -> tuple[str, int, float]:
        class_name = prediction.get("class", prediction.get("class_name", prediction.get("label", "")))
        return (
            str(class_name),
            int(prediction.get("class_id", 0)),
            DetectionsClassesReplacementExecutor._prediction_confidence(prediction),
        )

    @staticmethod
    def _replace_detection(
        detection: DetectionPrediction,
        replacement: tuple[str, int, float],
    ) -> DetectionPrediction:
        class_name, class_id, confidence = replacement
        detection_data = DetectionsClassesReplacementExecutor._model_to_dict(detection, by_alias=True)
        original_detection_id = detection_data.get("detection_id", "")
        metadata = deepcopy(detection_data.get("metadata", {}))
        metadata["original_detection_id"] = original_detection_id

        detection_data.update(
            {
                "detection_id": str(uuid4()),
                "class": class_name,
                "class_id": class_id,
                "confidence": confidence,
                "metadata": metadata,
            }
        )
        return DetectionsClassesReplacementExecutor._parse_detection(detection_data)

    @staticmethod
    def _parse_detection(value: dict[str, Any]) -> DetectionPrediction:
        if hasattr(DetectionPrediction, "model_validate"):
            return DetectionPrediction.model_validate(value)
        return DetectionPrediction.parse_obj(value)

    @staticmethod
    def _model_to_dict(model: Any, *, by_alias: bool = False) -> dict[str, Any]:
        if hasattr(model, "model_dump"):
            return model.model_dump(by_alias=by_alias)
        return model.dict(by_alias=by_alias)
