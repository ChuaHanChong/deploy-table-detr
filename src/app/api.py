"""Serving API."""
import io
import random
import sys
from datetime import datetime
from functools import wraps
from http import HTTPStatus
from typing import Callable, Dict

import numpy as np
import torch
from fastapi import FastAPI, File, Request, UploadFile
from PIL import Image

sys.path.extend(["src/table_detr/detr", "src/table_detr/src"])
from app.inference import load_artifacts, pipeline
from config.config import logger

# Define seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# Define application
app = FastAPI(
    title="Table Transformer Serving",
    version="0.1",
)


@app.on_event("startup")
def load_inference_artifacts():
    """Load saved artificats of Table-Transformer."""
    global detection_preprocessor, detection_model, structure_preprocessor, structure_model

    detection_preprocessor, detection_model = load_artifacts(
        data_type="detection",
        config_file="src/table_detr/src/detection_config.json",
        model_load_path="models/pubtables1m_detection_detr_r18.pth",
    )

    structure_preprocessor, structure_model = load_artifacts(
        data_type="structure",
        config_file="src/table_detr/src/structure_config.json",
        model_load_path="models/pubtables1m_structure_detr_r18.pth",
    )

    logger.info("Ready for inference!")


def construct_response(f):
    """Construct a JSON response for an endpoint."""

    @wraps(f)
    def wrap(request: Request, *args, **kwargs) -> Dict:
        results = f(request, *args, **kwargs)
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }
        if "data" in results:
            response["data"] = results["data"]
        return response

    return wrap


@app.post("/predict", tags=["Prediction"])
@construct_response
def _predict(request: Request, file: UploadFile = File(...)) -> Dict:
    """Predict tags for a list of texts."""
    contents = file.file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    table_records = pipeline(
        detection_preprocessor=detection_preprocessor,
        detection_model=detection_model,
        structure_preprocessor=structure_preprocessor,
        structure_model=structure_model,
        image=img,
    )

    response = {"message": HTTPStatus.OK.phrase, "status-code": HTTPStatus.OK, "data": {"records": table_records}}

    file.file.close()
    return response


def update_schema_name(app: FastAPI, function: Callable, name: str) -> None:
    """Update schema name.

    Updates the Pydantic schema name for a FastAPI function that takes
    in a fastapi.UploadFile = File(...) or bytes = File(...).

    This is a known issue that was reported on FastAPI#1442 in which
    the schema for file upload routes were auto-generated with no
    customization options. This renames the auto-generated schema to
    something more useful and clear.

    Args:
        app: The FastAPI application to modify.
        function: The function object to modify.
        name: The new name of the schema.
    """
    for route in app.routes:
        if route.endpoint is function:
            route.body_field.type_.__name__ = name
            break


update_schema_name(app, _predict, "PredictSchema")
