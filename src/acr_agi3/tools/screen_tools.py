"""On-demand, read-only visual access to the latest gateway observation."""

from __future__ import annotations

import base64
import io

import numpy as np

from acr_agi3.dsl.renderer import render_grid_to_image
from acr_agi3.harness.vision_observation import normalize_grid


class ScreenTools:
    def __init__(self) -> None:
        self.frames: dict[int, np.ndarray] = {}
        self.current_id = 0
        self.viewed: set[int] = set()

    def publish(self, grid) -> int:
        """Receive a frame without placing any pixels in the model context."""
        self.current_id += 1
        self.frames[self.current_id] = normalize_grid(grid).copy()
        for key in list(self.frames)[:-2]:
            del self.frames[key]
            self.viewed.discard(key)
        return self.current_id

    def observe_screen(
        self,
        view: str = "current",
        x: int = 0,
        y: int = 0,
        width: int = 0,
        height: int = 0,
    ) -> dict:
        """View current/previous/both frames, optionally cropped in game coordinates.

        Read-only: looking never consumes a game action. Returned images are
        delivered to the local VLM in its next inference, not just described.
        """
        if view not in ("current", "previous", "both") or not self.frames:
            return {"error": "Use current, previous or both with an available observation."}
        ids = sorted(self.frames)
        if view == "previous" and len(ids) < 2:
            return {"error": "No previous frame is available."}
        selected = ids if view == "both" else [ids[-1] if view == "current" else ids[-2]]
        images = []
        for frame_id in selected:
            grid = self.frames[frame_id]
            h, w = grid.shape
            cw, ch = width or w, height or h
            if x < 0 or y < 0 or cw <= 0 or ch <= 0 or x + cw > w or y + ch > h:
                return {"error": "Crop must lie inside every requested frame."}
            stream = io.BytesIO()
            render_grid_to_image(grid[y : y + ch, x : x + cw], cell_size=12).save(
                stream, format="PNG"
            )
            images.append(
                {
                    "frame_id": frame_id,
                    "origin": [x, y],
                    "grid_shape": [ch, cw],
                    "mime_type": "image/png",
                    "data": base64.b64encode(stream.getvalue()).decode(),
                }
            )
        self.viewed.update(selected)
        return {"screen_observation": {"images": images, "current_frame_id": self.current_id}}
