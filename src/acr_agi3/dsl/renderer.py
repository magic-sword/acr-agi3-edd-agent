"""ARC-AGI-3 グリッド画像レンダラー.

ARC 公式 10 色カラーパレットに基づき、
2次元数値グリッドをセル境界線付きのカラー画像 (PIL Image) に高速レンダリングします。
完全オフラインかつインメモリで動作します。
"""

from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

# ARC-AGI 公式 10 色カラーマップ (RGB)
# 0: 黒, 1: 青, 2: 赤, 3: 緑, 4: 黄, 5: 灰, 6: マゼンタ, 7: オレンジ, 8: 水色, 9: 茶色
ARC_COLORS = {
    0: (0, 0, 0),  # Black
    1: (0, 116, 217),  # Blue
    2: (255, 65, 54),  # Red
    3: (46, 204, 64),  # Green
    4: (255, 220, 0),  # Yellow
    5: (170, 170, 170),  # Gray
    6: (240, 18, 190),  # Magenta
    7: (255, 133, 27),  # Orange
    8: (127, 219, 255),  # Teal / Light Blue
    9: (135, 12, 37),  # Maroon / Brown
}

# グリッド線の色 (薄いグレー)
GRID_LINE_COLOR = (60, 60, 60)


def render_grid_to_image(
    grid: np.ndarray,
    cell_size: int = 24,
    grid_line_width: int = 1,
    scale: Optional[int] = None,
) -> Image.Image:
    """2次元グリッド配列をカラー画像 (PIL.Image) に変換する.

    Args:
        grid: (H, W) の 2D numpy 配列 (要素は 0〜9 の整数)
        cell_size: 1 セルあたりのピクセル幅・高さ
        grid_line_width: セル間の境界線の太さ (ピクセル)
        scale: cell_size のエイリアス

    Returns:
        RGB 形式の PIL Image
    """
    if scale is not None:
        cell_size = scale
    arr = np.array(grid, dtype=int)

    if arr.ndim != 2:
        raise ValueError(f"Grid must be 2-dimensional, got shape {arr.shape}")

    height, width = arr.shape
    img_width = width * cell_size
    img_height = height * cell_size

    image = Image.new("RGB", (img_width, img_height), color=GRID_LINE_COLOR)
    draw = ImageDraw.Draw(image)

    for r in range(height):
        for c in range(width):
            color_id = int(arr[r, c])
            color = ARC_COLORS.get(color_id, (0, 0, 0))

            x0 = c * cell_size
            y0 = r * cell_size
            x1 = x0 + cell_size - grid_line_width
            y1 = y0 + cell_size - grid_line_width

            draw.rectangle([x0, y0, x1, y1], fill=color)

    return image


def render_task_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    cell_size: int = 20,
    spacing: int = 40,
) -> Image.Image:
    """Input グリッドと Output グリッドを横に並べた対比画像を生成する.

    Args:
        input_grid: 入力グリッド
        output_grid: 期待される出力グリッド
        cell_size: セルサイズ
        spacing: 入出力間の余白幅

    Returns:
        横並びの比較 PIL Image
    """
    img_in = render_grid_to_image(input_grid, cell_size=cell_size)
    img_out = render_grid_to_image(output_grid, cell_size=cell_size)

    total_width = img_in.width + spacing + img_out.width
    max_height = max(img_in.height, img_out.height)

    # 余白込みのベース画像 (背景は濃いグレー)
    pair_img = Image.new("RGB", (total_width, max_height), color=(30, 30, 30))

    # 中央揃えで貼り付け
    y_in = (max_height - img_in.height) // 2
    y_out = (max_height - img_out.height) // 2

    pair_img.paste(img_in, (0, y_in))
    pair_img.paste(img_out, (img_in.width + spacing, y_out))

    # 中央に矢印または区切りを描画
    draw = ImageDraw.Draw(pair_img)
    mid_x = img_in.width + spacing // 2
    mid_y = max_height // 2
    draw.polygon(
        [
            (mid_x - 8, mid_y - 8),
            (mid_x + 8, mid_y),
            (mid_x - 8, mid_y + 8),
        ],
        fill=(200, 200, 200),
    )

    return pair_img
