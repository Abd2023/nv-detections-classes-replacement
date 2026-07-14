from __future__ import annotations

from typing import Any, Dict, List, Literal, Union

from pydantic import BaseModel, Field, root_validator


class NovaVisionModel(BaseModel):
    """Small Pydantic base model compatible with v1 and v2 runtimes."""

    class Config:
        extra = "forbid"
        allow_population_by_field_name = True
        populate_by_name = True


class DetectionPrediction(NovaVisionModel):
    detection_id: str = ""
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    class_name: str = Field(default="", alias="class")
    class_id: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


ClassificationPredictionValue = Union[Dict[str, Any], str, List[str]]


class ObjectDetectionPredictions(NovaVisionModel):
    name: Literal["inputDetections"] = "inputDetections"
    value: List[DetectionPrediction] = Field(default_factory=list)
    type: Literal["Detections"] = "Detections"
    field: Literal["hiddenInput"] = "hiddenInput"

    class Config:
        title = "Object Detection Predictions"


class ClassificationPredictions(NovaVisionModel):
    name: Literal["inputData"] = "inputData"
    value: List[ClassificationPredictionValue] = Field(default_factory=list)
    type: Literal["Detections"] = "Detections"
    field: Literal["hiddenInput"] = "hiddenInput"

    class Config:
        title = "Classification Predictions"


class Predictions(NovaVisionModel):
    name: Literal["outputDetections"] = "outputDetections"
    value: List[DetectionPrediction] = Field(default_factory=list)
    type: Literal["Detections"] = "Detections"
    field: Literal["hiddenInput"] = "hiddenInput"

    class Config:
        title = "Predictions"


class AutoMatchingModeOption(NovaVisionModel):
    name: Literal["Auto"] = "Auto"
    value: Literal[1] = 1
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Auto"


class ParentIdOnlyMatchingModeOption(NovaVisionModel):
    name: Literal["ParentIdOnly"] = "ParentIdOnly"
    value: Literal[2] = 2
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Parent ID Only"


class PositionalOnlyMatchingModeOption(NovaVisionModel):
    name: Literal["PositionalOnly"] = "PositionalOnly"
    value: Literal[3] = 3
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Positional Only"


class MatchingMode(NovaVisionModel):
    name: Literal["MatchingMode"] = "MatchingMode"
    value: Union[
        AutoMatchingModeOption,
        ParentIdOnlyMatchingModeOption,
        PositionalOnlyMatchingModeOption,
    ] = Field(default_factory=AutoMatchingModeOption)
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Matching Mode"


class FallbackClassName(NovaVisionModel):
    name: Literal["FallbackClassName"] = "FallbackClassName"
    value: str = ""
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Fallback Class Name"


class FallbackClassId(NovaVisionModel):
    name: Literal["FallbackClassId"] = "FallbackClassId"
    value: int = -1
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Fallback Class ID"


class DetectionsClassesReplacementInputs(NovaVisionModel):
    inputDetections: ObjectDetectionPredictions = Field(default_factory=ObjectDetectionPredictions)
    inputData: ClassificationPredictions = Field(default_factory=ClassificationPredictions)

    @root_validator(pre=True)
    def map_legacy_input_names(cls, values):
        if not isinstance(values, dict):
            return values

        values = dict(values)
        if "ObjectDetectionPredictions" in values and "inputDetections" not in values:
            values["inputDetections"] = values["ObjectDetectionPredictions"]
        if "ClassificationPredictions" in values and "inputData" not in values:
            values["inputData"] = values["ClassificationPredictions"]
        if "inputClassificationPredictions" in values and "inputData" not in values:
            values["inputData"] = values["inputClassificationPredictions"]
        values.pop("ObjectDetectionPredictions", None)
        values.pop("ClassificationPredictions", None)
        values.pop("inputClassificationPredictions", None)
        return values

    @property
    def object_detection_predictions(self) -> ObjectDetectionPredictions:
        return self.inputDetections

    @property
    def classification_predictions(self) -> ClassificationPredictions:
        return self.inputData

    @property
    def input_detections(self) -> ObjectDetectionPredictions:
        return self.inputDetections

    @property
    def input_classification_predictions(self) -> ClassificationPredictions:
        return self.inputData


class DetectionsClassesReplacementConfigs(NovaVisionModel):
    matching_mode: MatchingMode = Field(default_factory=MatchingMode, alias="MatchingMode")
    fallback_class_name: FallbackClassName = Field(
        default_factory=FallbackClassName,
        alias="FallbackClassName",
    )
    fallback_class_id: FallbackClassId = Field(
        default_factory=FallbackClassId,
        alias="FallbackClassId",
    )


class DetectionsClassesReplacementOutputs(NovaVisionModel):
    outputDetections: Predictions = Field(default_factory=Predictions)

    @root_validator(pre=True)
    def map_legacy_output_names(cls, values):
        if not isinstance(values, dict):
            return values

        values = dict(values)
        if "Predictions" in values and "outputDetections" not in values:
            values["outputDetections"] = values["Predictions"]
        values.pop("Predictions", None)
        return values

    @property
    def predictions(self) -> Predictions:
        return self.outputDetections

    @property
    def output_detections(self) -> Predictions:
        return self.outputDetections


class DetectionsClassesReplacementRequest(NovaVisionModel):
    name: Literal["DetectionsClassesReplacementExecutor"] = "DetectionsClassesReplacementExecutor"
    type: Literal["Request"] = "Request"
    inputs: DetectionsClassesReplacementInputs = Field(
        default_factory=DetectionsClassesReplacementInputs
    )
    configs: DetectionsClassesReplacementConfigs = Field(
        default_factory=DetectionsClassesReplacementConfigs
    )

    class Config:
        title = "Detections Classes Replacement Request"
        schema_extra = {"target": "configs"}
        json_schema_extra = {"target": "configs"}


class DetectionsClassesReplacementResponse(NovaVisionModel):
    name: Literal["DetectionsClassesReplacementExecutor"] = "DetectionsClassesReplacementExecutor"
    type: Literal["Response"] = "Response"
    outputs: DetectionsClassesReplacementOutputs = Field(
        default_factory=DetectionsClassesReplacementOutputs
    )

    class Config:
        title = "Detections Classes Replacement Response"


class DetectionsClassesReplacementExecutorOption(NovaVisionModel):
    name: Literal["DetectionsClassesReplacementExecutor"] = "DetectionsClassesReplacementExecutor"
    value: Union[
        DetectionsClassesReplacementRequest,
        DetectionsClassesReplacementResponse,
    ] = Field(default_factory=DetectionsClassesReplacementRequest)
    type: Literal["Executor"] = "Executor"
    field: Literal["option"] = "option"

    class Config:
        title = "Detections Classes Replacement Executor"
        schema_extra = {"target": {"value": 0}}
        json_schema_extra = {"target": {"value": 0}}


class ConfigExecutor(NovaVisionModel):
    name: Literal["Executor"] = "Executor"
    value: DetectionsClassesReplacementExecutorOption = Field(
        default_factory=DetectionsClassesReplacementExecutorOption
    )
    type: Literal["Executor"] = "Executor"
    field: Literal["option"] = "option"

    class Config:
        title = "Executor"
        schema_extra = {"target": "value"}
        json_schema_extra = {"target": "value"}


class PackageConfigs(NovaVisionModel):
    executor: ConfigExecutor = Field(default_factory=ConfigExecutor, alias="Executor")


class PackageModel(NovaVisionModel):
    name: Literal["DetectionsClassesReplacement"] = "DetectionsClassesReplacement"
    type: Literal["component"] = "component"
    configs: PackageConfigs = Field(default_factory=PackageConfigs)

    class Config:
        title = "Detections Classes Replacement"
