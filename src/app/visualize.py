"""Prediction visualization."""
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from table_detr.src.eval import get_bbox_decorations


def visualize_bbox(image, preds, data_type):
    """Visualize bbox predictions."""
    fig, ax = plt.subplots(1)
    ax.imshow(image, interpolation="lanczos")

    for pred in preds:
        label, bbox = pred["label"], pred["bbox"]

        color, alpha, linewidth, hatch = get_bbox_decorations(data_type, label)

        # Fill
        rect = patches.Rectangle(
            bbox[:2],
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=linewidth,
            alpha=alpha,
            edgecolor="none",
            facecolor=color,
            linestyle=None,
        )
        ax.add_patch(rect)

        # Hatch
        rect = patches.Rectangle(
            bbox[:2],
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=1,
            alpha=0.4,
            edgecolor=color,
            facecolor="none",
            linestyle="--",
            hatch=hatch,
        )
        ax.add_patch(rect)

        # Edge
        rect = patches.Rectangle(
            bbox[:2],
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=linewidth,
            edgecolor=color,
            facecolor="none",
            linestyle="--",
        )
        ax.add_patch(rect)

    fig.set_size_inches((15, 15))
    plt.axis("off")


def visualize_postprocessed_cell(image, cells):
    """Visualize postprocessed cells."""
    fig, ax = plt.subplots(1)
    ax.imshow(image, interpolation="lanczos")

    for cell in cells:
        bbox = cell["bbox"]
        if cell["header"]:
            alpha = 0.3
        else:
            alpha = 0.125

        rect = patches.Rectangle(
            bbox[:2],
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=1,
            edgecolor="none",
            facecolor="magenta",
            alpha=alpha,
        )
        ax.add_patch(rect)

        rect = patches.Rectangle(
            bbox[:2],
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=1,
            edgecolor="magenta",
            facecolor="none",
            linestyle="--",
            alpha=0.08,
            hatch="///",
        )
        ax.add_patch(rect)

        rect = patches.Rectangle(
            bbox[:2],
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=1,
            edgecolor="magenta",
            facecolor="none",
            linestyle="--",
        )
        ax.add_patch(rect)

    fig.set_size_inches((15, 15))
    plt.axis("off")
