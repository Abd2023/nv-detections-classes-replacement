from __future__ import annotations

from typing import Any, Dict, List, Literal, Union

from pydantic import BaseModel, Field, root_validator

try:
    from sdks.novavision.src.base.model import (  # type: ignore
        Configs as NovaVisionConfigs,
        Executor as NovaVisionExecutor,
        Inputs as NovaVisionInputs,
        Outputs as NovaVisionOutputs,
        Package as NovaVisionPackage,
        Param as NovaVisionParam,
        Request as NovaVisionRequest,
        Response as NovaVisionResponse,
    )
except ImportError:
    NovaVisionConfigs = BaseModel
    NovaVisionExecutor = BaseModel
    NovaVisionInputs = BaseModel
    NovaVisionOutputs = BaseModel
    NovaVisionPackage = BaseModel
    NovaVisionParam = BaseModel
    NovaVisionRequest = BaseModel
    NovaVisionResponse = BaseModel


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


class InputImage(NovaVisionParam):
    name: Literal["inputImage"] = "inputImage"
    value: Any = Field(default_factory=list)
    type: Literal["Images"] = "Images"
    field: Literal["input"] = "input"

    class Config:
        extra = "forbid"
        allow_population_by_field_name = True
        populate_by_name = True
        title = "Input Image"


class ObjectDetectionPredictions(NovaVisionParam):
    name: Literal["inputDetections"] = "inputDetections"
    value: List[DetectionPrediction] = Field(default_factory=list)
    type: Literal["Detections"] = "Detections"
    field: Literal["input"] = "input"

    class Config:
        extra = "forbid"
        allow_population_by_field_name = True
        populate_by_name = True
        title = "Object Detection Predictions"


class ClassificationPredictions(NovaVisionParam):
    name: Literal["inputData"] = "inputData"
    value: List[ClassificationPredictionValue] = Field(default_factory=list)
    type: Literal["Detections"] = "Detections"
    field: Literal["input"] = "input"

    class Config:
        extra = "forbid"
        allow_population_by_field_name = True
        populate_by_name = True
        title = "Classification Predictions"


class OutputImages(NovaVisionParam):
    name: Literal["outputImages"] = "outputImages"
    value: Any = Field(default_factory=list)
    type: Literal["Images"] = "Images"

    class Config:
        extra = "forbid"
        allow_population_by_field_name = True
        populate_by_name = True
        title = "Output Images"


class Predictions(NovaVisionParam):
    name: Literal["outputDetections"] = "outputDetections"
    value: List[DetectionPrediction] = Field(default_factory=list)
    type: Literal["Detections"] = "Detections"

    class Config:
        extra = "forbid"
        allow_population_by_field_name = True
        populate_by_name = True
        title = "Predictions"


class AutoMatchingModeOption(NovaVisionParam):
    name: Literal["Auto"] = "Auto"
    value: Literal[1] = 1
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        extra = "forbid"
        allow_population_by_field_name = True
        populate_by_name = True
        title = "Auto"


class ParentIdOnlyMatchingModeOption(NovaVisionParam):
    name: Literal["ParentIdOnly"] = "ParentIdOnly"
    value: Literal[2] = 2
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        extra = "forbid"
        allow_population_by_field_name = True
        populate_by_name = True
        title = "Parent ID Only"


class PositionalOnlyMatchingModeOption(NovaVisionParam):
    name: Literal["PositionalOnly"] = "PositionalOnly"
    value: Literal[3] = 3
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        extra = "forbid"
        allow_population_by_field_name = True
        populate_by_name = True
        title = "Positional Only"


class MatchingMode(NovaVisionParam):
    name: Literal["MatchingMode"] = "MatchingMode"
    value: Union[
        AutoMatchingModeOption,
        ParentIdOnlyMatchingModeOption,
        PositionalOnlyMatchingModeOption,
    ] = Field(default_factory=AutoMatchingModeOption)
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        extra = "forbid"
        allow_population_by_field_name = True
        populate_by_name = True
        title = "Matching Mode"


class FallbackClassName(NovaVisionParam):
    name: Literal["FallbackClassName"] = "FallbackClassName"
    value: str = ""
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"

    class Config:
        extra = "forbid"
        allow_population_by_field_name = True
        populate_by_name = True
        title = "Fallback Class Name"


class FallbackClassId(NovaVisionParam):
    name: Literal["FallbackClassId"] = "FallbackClassId"
    value: int = -1
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        extra = "forbid"
        allow_population_by_field_name = True
        populate_by_name = True
        title = "Fallback Class ID"


class DetectionsClassesReplacementInputs(NovaVisionInputs):
    inputImage: InputImage = Field(default_factory=InputImage)
    inputDetections: ObjectDetectionPredictions = Field(default_factory=ObjectDetectionPredictions)
    inputData: ClassificationPredictions = Field(default_factory=ClassificationPredictions)

    @root_validator(pre=True)
    def map_legacy_input_names(cls, values):
        if not isinstance(values, dict):
            return values

        values = dict(values)
        if "InputImage" in values and "inputImage" not in values:
            values["inputImage"] = values["InputImage"]
        if "ObjectDetectionPredictions" in values and "inputDetections" not in values:
            values["inputDetections"] = values["ObjectDetectionPredictions"]
        if "ClassificationPredictions" in values and "inputData" not in values:
            values["inputData"] = values["ClassificationPredictions"]
        if "inputClassificationPredictions" in values and "inputData" not in values:
            values["inputData"] = values["inputClassificationPredictions"]
        values.pop("InputImage", None)
        values.pop("ObjectDetectionPredictions", None)
        values.pop("ClassificationPredictions", None)
        values.pop("inputClassificationPredictions", None)
        return values

    @property
    def input_image(self) -> InputImage:
        return self.inputImage

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

    class Config:
        extra = "forbid"
        allow_population_by_field_name = True
        populate_by_name = True


class DetectionsClassesReplacementConfigs(NovaVisionConfigs):
    matching_mode: MatchingMode = Field(default_factory=MatchingMode, alias="MatchingMode")
    fallback_class_name: FallbackClassName = Field(
        default_factory=FallbackClassName,
        alias="FallbackClassName",
    )
    fallback_class_id: FallbackClassId = Field(
        default_factory=FallbackClassId,
        alias="FallbackClassId",
    )

    class Config:
        extra = "forbid"
        allow_population_by_field_name = True
        populate_by_name = True


class DetectionsClassesReplacementOutputs(NovaVisionOutputs):
    outputImages: OutputImages = Field(default_factory=OutputImages)
    outputDetections: Predictions = Field(default_factory=Predictions)

    @root_validator(pre=True)
    def map_legacy_output_names(cls, values):
        if not isinstance(values, dict):
            return values

        values = dict(values)
        if "OutputImages" in values and "outputImages" not in values:
            values["outputImages"] = values["OutputImages"]
        if "Predictions" in values and "outputDetections" not in values:
            values["outputDetections"] = values["Predictions"]
        values.pop("OutputImages", None)
        values.pop("Predictions", None)
        return values

    @property
    def output_images(self) -> OutputImages:
        return self.outputImages

    @property
    def predictions(self) -> Predictions:
        return self.outputDetections

    @property
    def output_detections(self) -> Predictions:
        return self.outputDetections

    class Config:
        extra = "forbid"
        allow_population_by_field_name = True
        populate_by_name = True


class DetectionsClassesReplacementRequest(NovaVisionRequest):
    name: Literal["DetectionsClassesReplacementExecutor"] = "DetectionsClassesReplacementExecutor"
    type: Literal["Request"] = "Request"
    inputs: DetectionsClassesReplacementInputs = Field(
        default_factory=DetectionsClassesReplacementInputs
    )
    configs: DetectionsClassesReplacementConfigs = Field(
        default_factory=DetectionsClassesReplacementConfigs
    )

    class Config:
        extra = "forbid"
        allow_population_by_field_name = True
        populate_by_name = True
        title = "Detections Classes Replacement Request"
        schema_extra = {"target": "configs"}
        json_schema_extra = {"target": "configs"}


class DetectionsClassesReplacementResponse(NovaVisionResponse):
    name: Literal["DetectionsClassesReplacementExecutor"] = "DetectionsClassesReplacementExecutor"
    type: Literal["Response"] = "Response"
    outputs: DetectionsClassesReplacementOutputs = Field(
        default_factory=DetectionsClassesReplacementOutputs
    )

    class Config:
        extra = "forbid"
        allow_population_by_field_name = True
        populate_by_name = True
        title = "Detections Classes Replacement Response"


class DetectionsClassesReplacementExecutorOption(NovaVisionExecutor):
    name: Literal["DetectionsClassesReplacementExecutor"] = "DetectionsClassesReplacementExecutor"
    value: Union[
        DetectionsClassesReplacementRequest,
        DetectionsClassesReplacementResponse,
    ] = Field(default_factory=DetectionsClassesReplacementRequest)
    type: Literal["Executor"] = "Executor"
    field: Literal["option"] = "option"

    class Config:
        extra = "forbid"
        allow_population_by_field_name = True
        populate_by_name = True
        title = "Detections Classes Replacement Executor"
        schema_extra = {"target": {"value": 0}}
        json_schema_extra = {"target": {"value": 0}}


class ConfigExecutor(NovaVisionExecutor):
    name: Literal["Executor"] = "Executor"
    value: DetectionsClassesReplacementExecutorOption = Field(
        default_factory=DetectionsClassesReplacementExecutorOption
    )
    type: Literal["Executor"] = "Executor"
    field: Literal["option"] = "option"

    class Config:
        extra = "forbid"
        allow_population_by_field_name = True
        populate_by_name = True
        title = "Executor"
        schema_extra = {"target": "value"}
        json_schema_extra = {"target": "value"}


class PackageConfigs(NovaVisionConfigs):
    executor: ConfigExecutor = Field(default_factory=ConfigExecutor, alias="Executor")

    class Config:
        extra = "forbid"
        allow_population_by_field_name = True
        populate_by_name = True


class PackageModel(NovaVisionPackage):
    name: Literal["DetectionsClassesReplacement"] = "DetectionsClassesReplacement"
    type: Literal["component"] = "component"
    configs: PackageConfigs = Field(default_factory=PackageConfigs)

    class Config:
        extra = "forbid"
        allow_population_by_field_name = True
        populate_by_name = True
        title = "Detections Classes Replacement"
