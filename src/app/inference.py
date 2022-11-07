"""Helper functions for model inference."""
import json
import uuid

import pandas as pd
import pytesseract
import torch

from table_detr.src.eval import (
    objects_to_cells,
    rescale_bboxes,
    structure_class_map,
    structure_class_names,
    structure_class_thresholds,
)
from table_detr.src.main import get_model, get_transform

MIN_SCORE = 0.5


def get_args(data_type, config_file, model_load_path):
    """Define arguments for table structure recognition model."""
    cmd_args = {
        "data_type": data_type,
        "config_file": config_file,
        "model_load_path": model_load_path,
    }
    config_args = json.load(open(cmd_args["config_file"], "rb"))
    config_args["device"] = "cpu"

    for key, value in cmd_args.items():
        if not key in config_args or not value is None:
            config_args[key] = value

    args = type("Args", (object,), config_args)
    return args


def load_artifacts(**kwargs):
    """Load saved artifacts (preprocessor & model)."""
    args = get_args(kwargs["data_type"], kwargs["config_file"], kwargs["model_load_path"])
    device = args.device

    model, _, _ = get_model(args, device)
    model.eval()

    preprocessor = get_transform(args.data_type, "val")

    return preprocessor, model


def postprocess(pred_logits, pred_bboxes, rescale_size):
    """Post-process the model predictions."""
    m = pred_logits.softmax(-1).max(-1)
    pred_labels = m.indices.tolist()
    pred_scores = m.values.tolist()
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, rescale_size)]

    return pred_labels, pred_scores, pred_bboxes


@torch.no_grad()
def predict(preprocessor, model, image):
    """Predict."""
    img_tensor, _ = preprocessor(image, {"boxes": []})

    outputs = model(img_tensor.unsqueeze(0))
    pred_logits, pred_bboxes = outputs["pred_logits"].squeeze(0), outputs["pred_boxes"].squeeze(0)

    return postprocess(pred_logits, pred_bboxes, image.size)


def select_table_predictions(pred_labels, pred_scores, pred_bboxes):
    """Select and reconstruct table predictions."""
    return [
        {"label": label, "score": score, "bbox": bbox}
        for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes)
        if not label > 1 and score > MIN_SCORE
    ]


def select_structure_predictions(pred_labels, pred_scores, pred_bboxes):
    """Select and reconstruct table predictions."""
    return [
        {"label": label, "score": score, "bbox": bbox}
        for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes)
        if not label > 5 and score > MIN_SCORE
    ]


def detect_text(image):
    """Detect text in image with Tesseract."""
    span_num = 0  # why?

    boxes = pytesseract.image_to_data(image)

    word_bbox = []
    for b in boxes.splitlines()[1:]:
        b = b.split()
        if len(b) == 12 and float(b[10]) > 0:
            x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])

            word_bbox.append(
                {
                    "bbox": [x, y, x + w, y + h],
                    "text": b[11],
                    "flags": 0,
                    "span_num": span_num,  # why?
                    "line_num": 0,  # why?
                    "block_num": 0,  # why?
                }
            )
            span_num += 1

    return word_bbox


def reconstruct_table(cells, orient="records"):
    """Reconstruct table into json format."""
    df = pd.DataFrame()
    for cell in cells:
        for col in cell["column_nums"]:
            for row in cell["row_nums"]:
                df.loc[row, col] = cell["cell_text"]

    df.sort_index(axis=1, inplace=True)
    df.sort_index(axis=0, inplace=True)

    return df.to_json(orient=orient)


def reconstruct_bbox(bbox):
    """Reconstruct the location of bounding box."""
    return {"Width": bbox[0], "Height": bbox[1], "Left": bbox[2], "Top": bbox[3]}


def pipeline(**kwargs):
    """Run full prediction pipeline."""
    table_records = {}

    # Predict table
    for i, img in enumerate(kwargs["images"]):
        pred_labels, pred_scores, pred_bboxes = predict(
            kwargs["detection_preprocessor"], kwargs["detection_model"], img
        )
        pred_tables = select_table_predictions(pred_labels, pred_scores, pred_bboxes)

        for pred in pred_tables:
            table_records[str(uuid.uuid4())] = {"Page": i, "Score": pred["score"], "BoundingBox": pred["bbox"]}

    # Reconstruct table
    for record in table_records.values():
        bbox = record["BoundingBox"]
        table_img = kwargs["images"][record["Page"]].crop(bbox)

        tokens = detect_text(table_img)

        pred_labels, pred_scores, pred_bboxes = predict(
            kwargs["structure_preprocessor"], kwargs["structure_model"], table_img
        )

        _, pred_cells, _ = objects_to_cells(
            pred_bboxes,
            pred_labels,
            pred_scores,
            tokens,
            structure_class_names,
            structure_class_thresholds,
            structure_class_map,
        )

        record["Table"] = reconstruct_table(pred_cells)

    # Reconstruct bounding box
    for record in table_records.values():
        record["BoundingBox"] = reconstruct_bbox(record["BoundingBox"])

    return table_records
