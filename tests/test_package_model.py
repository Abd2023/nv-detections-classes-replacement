from typing import get_args

import importlib

from src.models import (
    ClassificationPredictions,
    DetectionPrediction,
    DetectionsClassesReplacementRequest,
    DetectionsClassesReplacementResponse,
    MatchingMode,
    OutputImages,
    PackageModel,
    Predictions,
)
from src.models.PackageModel import PackageModel as SuitePackageModel


def model_fields(model_class):
    if hasattr(model_class, "model_fields"):
        return model_class.model_fields
    return model_class.__fields__


def model_to_dict(model, *, by_alias=False):
    if hasattr(model, "model_dump"):
        return model.model_dump(by_alias=by_alias)
    return model.dict(by_alias=by_alias)


def parse_model(model_class, value):
    if hasattr(model_class, "model_validate"):
        return model_class.model_validate(value)
    return model_class.parse_obj(value)


def model_schema(model_class):
    if hasattr(model_class, "model_json_schema"):
        return model_class.model_json_schema()
    return model_class.schema()


def test_package_model_imports_and_uses_component_type():
    package = PackageModel()

    assert SuitePackageModel is PackageModel
    assert package.name == "DetectionsClassesReplacement"
    assert package.type == "component"
    assert package.executor.name == "executor"
    assert package.executor.type == "executor"
    assert package.executor.field == "dependentDropdownlist"
    assert package.executor.value.name == "DetectionsClassesReplacementExecutor"
    assert package.executor.value.field == "executor"
    assert package.configs.executor.value.name == "DetectionsClassesReplacementExecutor"


def test_package_model_root_targets_executor_for_suite_form_renderer():
    schema = model_schema(PackageModel)

    assert schema["target"] == "executor"
    assert {"executor", "field"}.issubset(schema["properties"])
    assert "configs" not in schema["properties"]
    assert PackageModel(configs={"Executor": {}}).executor.name == "executor"


def test_suite_compatibility_module_exports_socket_classes():
    suite_module = importlib.import_module("src.models.PackageModel")

    assert suite_module.PackageModel is PackageModel
    assert suite_module.InputImage.__name__ == "InputImage"
    assert suite_module.ObjectDetectionPredictions.__name__ == "ObjectDetectionPredictions"
    assert suite_module.ClassificationPredictions.__name__ == "ClassificationPredictions"
    assert suite_module.OutputImages.__name__ == "OutputImages"
    assert suite_module.Predictions.__name__ == "Predictions"


def test_request_defaults_define_inputs_and_configs():
    request = DetectionsClassesReplacementRequest()

    assert request.name == "DetectionsClassesReplacementExecutor"
    assert request.type == "Request"
    assert request.inputs.inputImage.name == "inputImage"
    assert request.inputs.inputDetections.name == "inputDetections"
    assert request.inputs.inputData.name == "inputData"
    assert request.inputs.inputImage.field == "input"
    assert request.inputs.inputDetections.field == "input"
    assert request.inputs.inputData.field == "input"
    assert request.inputs.field == "input"
    assert request.inputs.type == "object"
    assert request.inputs.input_image.value == []
    assert request.inputs.input_image.type == "Images"
    assert request.inputs.object_detection_predictions.value == []
    assert request.inputs.object_detection_predictions.type == "Detections"
    assert request.inputs.classification_predictions.value == []
    assert request.configs.matching_mode.value.name == "Auto"
    assert request.configs.fallback_class_name.value == ""
    assert request.configs.fallback_class_id.value == -1


def test_matching_mode_exposes_three_options():
    options = get_args(model_fields(MatchingMode)["value"].annotation)

    assert {option.__name__ for option in options} == {
        "AutoMatchingModeOption",
        "ParentIdOnlyMatchingModeOption",
        "PositionalOnlyMatchingModeOption",
    }


def test_detection_prediction_supports_roboflow_style_fields():
    detection = parse_model(
        DetectionPrediction,
        {
            "detection_id": "det-1",
            "x": 100.0,
            "y": 80.0,
            "width": 40.0,
            "height": 30.0,
            "confidence": 0.75,
            "class": "vehicle",
            "class_id": 1,
            "metadata": {"source": "detector"},
        },
    )

    serialized = model_to_dict(detection, by_alias=True)

    assert detection.detection_id == "det-1"
    assert detection.class_name == "vehicle"
    assert serialized["class"] == "vehicle"
    assert serialized["metadata"] == {"source": "detector"}


def test_classification_predictions_accept_dicts_strings_and_string_lists():
    predictions = ClassificationPredictions(
        value=[
            {"parent_id": "det-1", "top": "truck", "confidence": 0.91, "class_id": 7},
            "bus",
            ["plate-123", "plate-124"],
        ]
    )

    assert predictions.name == "inputData"
    assert predictions.type == "Detections"
    assert predictions.value[0]["parent_id"] == "det-1"
    assert predictions.value[1] == "bus"
    assert predictions.value[2] == ["plate-123", "plate-124"]


def test_response_outputs_predictions():
    detection = DetectionPrediction(
        detection_id="new-det-1",
        x=100.0,
        y=80.0,
        width=40.0,
        height=30.0,
        confidence=0.91,
        class_name="truck",
        class_id=7,
        metadata={"original_detection_id": "det-1"},
    )
    response = DetectionsClassesReplacementResponse(
        outputs={
            "OutputImages": OutputImages(value=[{"uID": "image-1"}]),
            "Predictions": Predictions(value=[detection]),
        }
    )

    assert response.name == "DetectionsClassesReplacementExecutor"
    assert response.type == "Response"
    assert response.outputs.outputImages.name == "outputImages"
    assert response.outputs.outputImages.field == "data"
    assert response.outputs.output_images.value == [{"uID": "image-1"}]
    assert response.outputs.outputDetections.name == "outputDetections"
    assert response.outputs.outputDetections.field == "data"
    assert response.outputs.field == "output"
    assert response.outputs.type == "object"
    assert len(response.outputs.predictions.value) == 1
    assert response.outputs.predictions.value[0].class_name == "truck"


def test_request_and_response_serialize_with_canvas_socket_names():
    request = DetectionsClassesReplacementRequest()
    response = DetectionsClassesReplacementResponse()

    request_payload = model_to_dict(request, by_alias=True)
    response_payload = model_to_dict(response, by_alias=True)

    assert {"inputData", "inputDetections", "inputImage"}.issubset(request_payload["inputs"])
    assert request_payload["inputs"]["inputImage"]["name"] == "inputImage"
    assert request_payload["inputs"]["inputDetections"]["name"] == "inputDetections"
    assert request_payload["inputs"]["inputData"]["name"] == "inputData"
    assert request_payload["inputs"]["field"] == "input"
    assert {"outputDetections", "outputImages"}.issubset(response_payload["outputs"])
    assert response_payload["outputs"]["outputImages"]["name"] == "outputImages"
    assert response_payload["outputs"]["outputDetections"]["name"] == "outputDetections"
    assert response_payload["outputs"]["field"] == "output"
