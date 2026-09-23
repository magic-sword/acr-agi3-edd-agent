"""Game-frame grounding and deterministic necessary conditions for visual reviews.

An ROI is the effect region, including an expected destination when objects move.
It is not a semantic detector: pixel changes never prove a hypothesized relation.
"""

from copy import deepcopy

import numpy as np


def ground_target(target: dict, frame: np.ndarray, frame_id: int) -> dict:
    if not isinstance(target, dict):
        raise ValueError("target must describe an observed region.")
    region = [target.get(k) for k in ("x", "y", "width", "height")]
    if any(type(v) is not int for v in region):
        raise ValueError("target needs integer x, y, width, height in original coordinates.")
    x, y, width, height = region
    h, w = frame.shape
    if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > w or y + height > h:
        raise ValueError("target region must lie inside the observed frame.")
    if not isinstance(target.get("description"), str) or not target["description"].strip():
        raise ValueError(
            "Describe the visible target and expected effect region, not just coordinates."
        )
    return {k: deepcopy(target[k]) for k in ("x", "y", "width", "height", "description")} | {
        "frame_id": frame_id
    }


def compare_target(before: np.ndarray, after: np.ndarray, target: dict) -> dict:
    if before.shape != after.shape:
        return {"status": "incomparable", "reason": "frame_shape_changed"}
    x, y, width, height = (target[k] for k in ("x", "y", "width", "height"))
    mask = before != after
    region_changes = int(np.count_nonzero(mask[y : y + height, x : x + width]))
    return {
        "status": "comparable",
        "target_changed": bool(region_changes),
        "target_changed_pixels": region_changes,
        "screen_changed_pixels": int(np.count_nonzero(mask)),
        "semantic_effect_verified": False,
    }
