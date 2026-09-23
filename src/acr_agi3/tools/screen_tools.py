"""On-demand, read-only visual access to the latest gateway observation."""

from __future__ import annotations

import base64
import io

import numpy as np

from acr_agi3.dsl.renderer import GAME_COLORS, render_grid_to_image
from acr_agi3.harness.vision_observation import normalize_grid


class ScreenTools:
    def __init__(self) -> None:
        self.frames: dict[int, np.ndarray] = {}
        self.current_id = 0
        self.viewed: set[int] = set()
        self.viewed_regions: dict[int, list[tuple[int, int, int, int]]] = {}
        self.cursor: tuple[int, int] | None = None
        self.cursor_revision = 0
        self.cursor_observed: tuple[int, int] | None = None

    def clear_cursor(self) -> None:
        self.cursor = None
        self.cursor_revision += 1
        self.cursor_observed = None

    def move_cursor(self, x: int, y: int) -> dict:
        if not self.frames:
            raise ValueError("No current frame is available.")
        h, w = self.frames[self.current_id].shape
        if type(x) is not int or type(y) is not int or not (0 <= x < w and 0 <= y < h):
            raise ValueError("Cursor coordinates must be integers inside the current frame.")
        self.cursor = (x, y)
        self.cursor_revision += 1
        self.cursor_observed = None
        return {
            "cursor": [x, y],
            "cursor_revision": self.cursor_revision,
            "frame_id": self.current_id,
            "requires_observation": True,
        }

    def require_observed_cursor(self) -> tuple[int, int]:
        if self.cursor is None or self.cursor_observed != (self.current_id, self.cursor_revision):
            raise ValueError(
                "Move the cursor, then observe its reticle on the current frame before clicking."
            )
        return self.cursor

    def publish(self, grid) -> int:
        """Receive a frame without placing any pixels in the model context."""
        self.current_id += 1
        self.cursor_observed = None
        self.frames[self.current_id] = normalize_grid(grid).copy()
        for key in list(self.frames)[:-2]:
            del self.frames[key]
            self.viewed.discard(key)
            self.viewed_regions.pop(key, None)
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
        cursor_visible = False
        for frame_id in selected:
            grid = self.frames[frame_id]
            h, w = grid.shape
            cw, ch = width or w, height or h
            if x < 0 or y < 0 or cw <= 0 or ch <= 0 or x + cw > w or y + ch > h:
                return {"error": "Crop must lie inside every requested frame."}
            stream = io.BytesIO()
            reticle = None
            if self.cursor is not None:
                cx, cy = self.cursor
                if x <= cx < x + cw and y <= cy < y + ch:
                    reticle = (cx - x, cy - y)
                    cursor_visible = cursor_visible or frame_id == self.current_id
            render_grid_to_image(
                grid[y : y + ch, x : x + cw],
                cell_size=12,
                cursor_pos=reticle,
                palette=GAME_COLORS,
                grid_line_width=0,
            ).save(stream, format="PNG")
            images.append(
                {
                    "frame_id": frame_id,
                    "origin": [x, y],
                    "grid_shape": [ch, cw],
                    "cursor": list(self.cursor) if reticle is not None else None,
                    "cursor_revision": self.cursor_revision if reticle is not None else None,
                    "palette": "arc-agi-3",
                    "reticle_is_annotation": reticle is not None,
                    "mime_type": "image/png",
                    "data": base64.b64encode(stream.getvalue()).decode(),
                }
            )
        self.viewed.update(selected)
        for item in images:
            ox, oy = item["origin"]
            rh, rw = item["grid_shape"]
            self.viewed_regions.setdefault(item["frame_id"], []).append((ox, oy, rw, rh))
        if cursor_visible:
            self.cursor_observed = (self.current_id, self.cursor_revision)
        return {"screen_observation": {"images": images, "current_frame_id": self.current_id}}

    def require_viewed_region(self, frame_id: int, target: dict) -> None:
        tx, ty, tw, th = (target[k] for k in ("x", "y", "width", "height"))
        if not any(
            x <= tx and y <= ty and tx + tw <= x + w and ty + th <= y + h
            for x, y, w, h in self.viewed_regions.get(frame_id, [])
        ):
            raise ValueError("Observe the entire predicted target region in this frame first.")
