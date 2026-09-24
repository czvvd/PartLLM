"""Shared input protocol for interactive mask refinement.

Refine samples deliberately discard the original RGB signal.  The current
mask is encoded as white foreground on a gray background, while click history
is represented as accumulated positive and negative coordinate sets.
"""

from collections.abc import Mapping, Sequence

import numpy as np


REFINE_BACKGROUND_VALUE = 128
REFINE_FOREGROUND_VALUE = 255


def encode_refine_mask_colors(input_colors, foreground_mask):
    """Return a copy of ``input_colors`` with a binary mask RGB encoding.

    The first three channels are replaced with gray (background) or white
    (foreground).  Any channels after RGB, such as alpha, are preserved.
    ``input_colors`` itself is never modified.
    """
    colors = np.asarray(input_colors)
    if colors.ndim != 2 or colors.shape[1] < 3:
        raise ValueError(
            f"input_colors must have shape [N, C] with C >= 3, got {colors.shape}"
        )
    if not (
        np.issubdtype(colors.dtype, np.integer)
        or np.issubdtype(colors.dtype, np.floating)
    ):
        raise TypeError(f"input_colors must be numeric, got dtype={colors.dtype}")
    if np.issubdtype(colors.dtype, np.integer):
        limits = np.iinfo(colors.dtype)
        if limits.min > REFINE_BACKGROUND_VALUE or limits.max < REFINE_FOREGROUND_VALUE:
            raise ValueError(
                f"input_colors dtype={colors.dtype} cannot represent "
                f"{REFINE_BACKGROUND_VALUE}..{REFINE_FOREGROUND_VALUE}"
            )

    mask = np.asarray(foreground_mask)
    if mask.shape != (colors.shape[0],):
        raise ValueError(
            f"foreground_mask must have shape ({colors.shape[0]},), got {mask.shape}"
        )
    if mask.dtype != np.bool_:
        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError("foreground_mask must contain only boolean/0/1 values")
        mask = mask.astype(bool)

    encoded = colors.copy()
    encoded[:, :3] = REFINE_BACKGROUND_VALUE
    encoded[mask, :3] = REFINE_FOREGROUND_VALUE
    return encoded


def _click_label(click: Mapping) -> int:
    if "label" in click:
        raw_label = click["label"]
        try:
            label = int(raw_label)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid click label: {raw_label!r}") from exc
        if label not in (0, 1):
            raise ValueError(f"click label must be 0 or 1, got {raw_label!r}")
        return label

    prompt_type = str(click.get("prompt_type", "")).strip().lower()
    if prompt_type in {"include", "positive", "promptable"}:
        return 1
    if prompt_type in {"exclude", "negative"}:
        return 0
    raise ValueError("each click must provide label=0/1 or a known prompt_type")


def _click_point(click: Mapping) -> tuple[float, float, float]:
    if "point" not in click:
        raise ValueError("each click must provide a point")
    point = np.asarray(click["point"], dtype=np.float64)
    if point.shape != (3,):
        raise ValueError(f"click point must have shape (3,), got {point.shape}")
    if not np.isfinite(point).all():
        raise ValueError("click point coordinates must be finite")
    return tuple(float(value) for value in point)


def _format_coordinate_set(points: Sequence[tuple[float, float, float]]) -> str:
    return "[" + ", ".join(
        f"({x:.3f}, {y:.3f}, {z:.3f})" for x, y, z in points
    ) + "]"


def format_refine_prompt(click_history, *, semantic) -> str:
    """Format all clicks as accumulated positive/negative coordinate sets.

    Args:
        click_history: Ordered mappings with ``point`` and either ``label``
            (1=positive, 0=negative) or ``prompt_type``.
        semantic: ``True``/``"semantic"`` asks the model to name the mask;
            ``False``/``"nosem"`` requests geometry-only segmentation.
    """
    if isinstance(semantic, str):
        mode = semantic.strip().lower()
        if mode not in {"semantic", "nosem"}:
            raise ValueError(f"semantic mode must be 'semantic' or 'nosem', got {semantic!r}")
        semantic = mode == "semantic"
    elif not isinstance(semantic, (bool, np.bool_)):
        raise TypeError("semantic must be a bool, 'semantic', or 'nosem'")

    clicks = list(click_history)
    if not clicks:
        raise ValueError("click_history must contain at least one click")

    positive_points = []
    negative_points = []
    for click in clicks:
        if not isinstance(click, Mapping):
            raise TypeError("each click_history item must be a mapping")
        point = _click_point(click)
        if _click_label(click) == 1:
            positive_points.append(point)
        else:
            negative_points.append(point)

    suffix = (
        "Please refine and name the mask."
        if semantic
        else "Please refine the geometric mask."
    )
    return (
        "The highlighted points in <point_cloud> show a segmentation mask. "
        f"Positive coordinates: {_format_coordinate_set(positive_points)}. "
        f"Negative coordinates: {_format_coordinate_set(negative_points)}. "
        "The positive coordinates should be included, and the negative "
        f"coordinates should be excluded. {suffix}"
    )

