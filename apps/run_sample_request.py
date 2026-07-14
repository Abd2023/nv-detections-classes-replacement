from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import urllib.request
import warnings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_REQUEST_PATH = PROJECT_ROOT / "apps" / "sample_request.json"

sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore", category=UserWarning, module=r"pydantic\..*")
warnings.filterwarnings("ignore", message=r".*PydanticDeprecatedSince20.*")

from src.executors import DetectionsClassesReplacementExecutor  # noqa: E402


def main() -> None:
    payload = load_sample_request()
    api_url = os.environ.get("NOVAVISION_API_URL", "").strip()

    if api_url:
        response_payload = post_to_api(payload, api_url)
        print_json("API response", response_payload)
        return

    response_payload = run_locally(payload)
    print_class_replacement_summary(payload, response_payload)
    print_json("Updated predictions", response_payload["outputs"]["Predictions"]["value"])


def load_sample_request() -> dict:
    return json.loads(SAMPLE_REQUEST_PATH.read_text(encoding="utf-8"))


def run_locally(payload: dict) -> dict:
    response = DetectionsClassesReplacementExecutor(payload).run()
    return model_to_dict(response, by_alias=True)


def post_to_api(payload: dict, api_url: str) -> dict:
    endpoint = api_url.rstrip("/")
    if not endpoint.endswith("/api"):
        endpoint = f"{endpoint}/api"

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def print_class_replacement_summary(request_payload: dict, response_payload: dict) -> None:
    input_detections = request_payload["inputs"]["ObjectDetectionPredictions"]["value"]
    output_detections = response_payload["outputs"]["Predictions"]["value"]

    print("Detections Classes Replacement sample")
    print("-------------------------------------")
    for original, updated in zip(input_detections, output_detections):
        print(
            f"{original['detection_id']}: "
            f"{original['class']} -> {updated['class']} "
            f"(confidence {updated['confidence']})"
        )
    print()


def print_json(title: str, payload: object) -> None:
    print(f"{title}:")
    print(json.dumps(payload, indent=2))


def model_to_dict(model: object, *, by_alias: bool = False) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump(by_alias=by_alias)
    return model.dict(by_alias=by_alias)


if __name__ == "__main__":
    main()

