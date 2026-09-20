"""ARC-AGI-3 グリッド画像レンダラー.

ARC 公式 10 色カラーパレットに基づき、
2次元数値グリッドをセル境界線付きのカラー画像 (PIL Image) に高速レンダリングします。
完全オフラインかつインメモリで動作します。
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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

ARC_COLOR_NAMES = {
    0: "Black",
    1: "Blue",
    2: "Red",
    3: "Green",
    4: "Yellow",
    5: "Gray",
    6: "Magenta",
    7: "Orange",
    8: "Teal",
    9: "Brown",
}


# グリッド線の色 (薄いグレー)
GRID_LINE_COLOR = (60, 60, 60)


def render_grid_to_image(
    grid: np.ndarray,
    cell_size: int = 24,
    grid_line_width: int = 1,
    scale: Optional[int] = None,
    cursor_pos: Optional[Tuple[int, int]] = None,
) -> Image.Image:
    """2次元グリッド配列をカラー画像 (PIL.Image) に変換する.

    Args:
        grid: (H, W) の 2D numpy 配列 (要素は 0〜9 の整数)
        cell_size: 1 セルあたりのピクセル幅・高さ
        grid_line_width: セル間の境界線の太さ (ピクセル)
        scale: cell_size のエイリアス
        cursor_pos: (col, row) のマウスカーソル表示座標

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

    # マウスカーソル (照準レティクル [ + ]) の重畳描画
    if cursor_pos is not None:
        cx, cy = cursor_pos
        if 0 <= cx < width and 0 <= cy < height:
            x0 = cx * cell_size
            y0 = cy * cell_size
            x1 = x0 + cell_size - 1
            y1 = y0 + cell_size - 1

            bracket_len = max(3, cell_size // 3)
            reticle_color = (255, 255, 255)
            shadow_color = (0, 0, 0)

            # 四隅のブラケット [ ] (外枠シャドウ付きで下地色問わず視認可能)
            for offset, col in [((1, 1), shadow_color), ((0, 0), reticle_color)]:
                dx, dy = offset
                # Top-Left
                draw.line([(x0 + dx, y0 + dy), (x0 + bracket_len + dx, y0 + dy)], fill=col, width=2)
                draw.line([(x0 + dx, y0 + dy), (x0 + dx, y0 + bracket_len + dy)], fill=col, width=2)
                # Top-Right
                draw.line([(x1 + dx, y0 + dy), (x1 - bracket_len + dx, y0 + dy)], fill=col, width=2)
                draw.line([(x1 + dx, y0 + dy), (x1 + dx, y0 + bracket_len + dy)], fill=col, width=2)
                # Bottom-Left
                draw.line([(x0 + dx, y1 + dy), (x0 + bracket_len + dx, y1 + dy)], fill=col, width=2)
                draw.line([(x0 + dx, y1 + dy), (x0 + dx, y1 - bracket_len + dy)], fill=col, width=2)
                # Bottom-Right
                draw.line([(x1 + dx, y1 + dy), (x1 - bracket_len + dx, y1 + dy)], fill=col, width=2)
                draw.line([(x1 + dx, y1 + dy), (x1 + dx, y1 - bracket_len + dy)], fill=col, width=2)

            # セル中央のターゲットドット/小さなクロスヘア (黄色 #FFFF00)
            mid_x = (x0 + x1) // 2
            mid_y = (y0 + y1) // 2
            draw.line([(mid_x - 3, mid_y), (mid_x + 3, mid_y)], fill=(255, 255, 0), width=1)
            draw.line([(mid_x, mid_y - 3), (mid_x, mid_y + 3)], fill=(255, 255, 0), width=1)

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


def _normalize_action_id(action: Optional[Union[int, str, Any]]) -> Optional[int]:
    """アクション名または数値を整数 ID (0〜7) に正規化."""
    if action is None:
        return None
    if isinstance(action, int):
        return action
    if hasattr(action, "value"):
        try:
            return int(action.value)
        except (ValueError, TypeError):
            pass

    import re
    act_str = str(action).strip().upper()
    # "ACTION1" や "UP (ACTION1)" から数字IDを抽出
    m = re.search(r"ACTION(\d+)", act_str)
    if m:
        return int(m.group(1))

    name_map = {
        "RESET": 0,
        "UP": 1,
        "DOWN": 2,
        "LEFT": 3,
        "RIGHT": 4,
        "ACTION1": 1,
        "ACTION2": 2,
        "ACTION3": 3,
        "ACTION4": 4,
        "ACTION5": 5,
        "ACTION6": 6,
        "ACTION7": 7,
        "CLICK": 6,
    }
    for k, v in name_map.items():
        if k in act_str:
            return v
    try:
        return int(act_str)
    except ValueError:
        return None


def render_gamepad_panel(
    width: int = 420,
    height: int = 140,
    available_actions: Optional[List[Union[int, str]]] = None,
    last_action: Optional[Union[int, str, Any]] = None,
    step_index: int = 0,
    game_state: str = "PLAYING",
) -> Image.Image:
    """ゲームボーイ / レトロコンソール風のコントローラー・HUDパネルをレンダリングする.

    Args:
        width: パネル幅 (ピクセル)
        height: パネル高さ (ピクセル)
        available_actions: 現在利用可能なアクションIDまたは名前リスト
        last_action: 直前ターンで実行されたアクション
        step_index: 現在のステップ番号
        game_state: ゲーム状態文字列 ("PLAYING", "WON", "LOST" など)

    Returns:
        RGB PIL Image
    """
    panel_img = Image.new("RGB", (width, height), color=(24, 27, 34))
    draw = ImageDraw.Draw(panel_img)
    font = ImageFont.load_default()

    # 外枠・ベゼル装飾
    draw.rectangle([1, 1, width - 2, height - 2], outline=(48, 54, 66), width=1)

    # 利用可能アクション ID の正規化集合
    if available_actions is None:
        avail_set = {1, 2, 3, 4, 6}
    else:
        avail_set = set()
        for a in available_actions:
            aid = _normalize_action_id(a)
            if aid is not None:
                avail_set.add(aid)

    last_aid = _normalize_action_id(last_action)

    def get_button_style(act_id: int) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]:
        """(fill_color, outline_color, text_color) を返却."""
        if act_id == last_aid:
            # 直前に実行されたアクション: 鮮やかなアンバー・シアン発光ハイライト
            return (180, 110, 15), (255, 215, 0), (255, 255, 255)
        elif act_id in avail_set:
            # 有効・選択可能: 明るいグレー・白
            return (45, 52, 64), (130, 145, 165), (230, 235, 245)
        else:
            # 無効 (Disabled): 暗い沈んだグレー
            return (22, 25, 30), (45, 50, 60), (75, 85, 95)

    # -------------------------------------------------------------
    # 1. 左側: 十字キー (D-Pad: UP[1], DOWN[2], LEFT[3], RIGHT[4])
    # -------------------------------------------------------------
    cx, cy = 65, 70
    arm_w, arm_l = 22, 26
    half_w = arm_w // 2

    # D-pad ベース十字の背景
    draw.rectangle([cx - half_w - 2, cy - arm_l - half_w - 2, cx + half_w + 2, cy + arm_l + half_w + 2], fill=(15, 17, 22))
    draw.rectangle([cx - arm_l - half_w - 2, cy - half_w - 2, cx + arm_l + half_w + 2, cy + half_w + 2], fill=(15, 17, 22))

    # UP (1)
    up_fill, up_out, up_txt = get_button_style(1)
    draw.rectangle([cx - half_w, cy - arm_l - half_w, cx + half_w, cy - half_w], fill=up_fill, outline=up_out)
    draw.polygon([(cx, cy - arm_l - 4), (cx - 6, cy - half_w - 4), (cx + 6, cy - half_w - 4)], fill=up_txt)

    # DOWN (2)
    dn_fill, dn_out, dn_txt = get_button_style(2)
    draw.rectangle([cx - half_w, cy + half_w, cx + half_w, cy + arm_l + half_w], fill=dn_fill, outline=dn_out)
    draw.polygon([(cx, cy + arm_l + 4), (cx - 6, cy + half_w + 4), (cx + 6, cy + half_w + 4)], fill=dn_txt)

    # LEFT (3)
    lt_fill, lt_out, lt_txt = get_button_style(3)
    draw.rectangle([cx - arm_l - half_w, cy - half_w, cx - half_w, cy + half_w], fill=lt_fill, outline=lt_out)
    draw.polygon([(cx - arm_l - 4, cy), (cx - half_w - 4, cy - 6), (cx - half_w - 4, cy + 6)], fill=lt_txt)

    # RIGHT (4)
    rt_fill, rt_out, rt_txt = get_button_style(4)
    draw.rectangle([cx + half_w, cy - half_w, cx + arm_l + half_w, cy + half_w], fill=rt_fill, outline=rt_out)
    draw.polygon([(cx + arm_l + 4, cy), (cx + half_w + 4, cy - 6), (cx + half_w + 4, cy + 6)], fill=rt_txt)

    # D-Pad 中央キャップ
    draw.rectangle([cx - half_w, cy - half_w, cx + half_w, cy + half_w], fill=(30, 35, 45), outline=(60, 70, 85))
    draw.text((cx - 15, cy + arm_l + half_w + 5), "D-PAD", fill=(130, 145, 165), font=font)

    # -------------------------------------------------------------
    # 2. 中央: HUD / LCD ディスプレイ (ステータス & リセットボタン)
    # -------------------------------------------------------------
    hud_x1 = max(130, cx + arm_l + half_w + 20)
    hud_x2 = min(width - 150, width - 130)
    if hud_x2 > hud_x1:
        # LCD 背景
        draw.rectangle([hud_x1, 14, hud_x2, 86], fill=(12, 16, 22), outline=(40, 50, 65), width=1)

        # ステータステキスト
        draw.text((hud_x1 + 8, 20), f"STEP: {step_index:03d}", fill=(0, 215, 255), font=font)
        draw.text((hud_x1 + 8, 36), f"STATE: {game_state.upper()}", fill=(160, 240, 160), font=font)

        last_str = "NONE"
        if last_aid is not None:
            id_to_name = {0: "RESET", 1: "UP (1)", 2: "DOWN (2)", 3: "LEFT (3)", 4: "RIGHT (4)", 5: "ACT5 (5)", 6: "CLICK (6)", 7: "ACT7 (7)"}
            last_str = id_to_name.get(last_aid, f"ACT {last_aid}")
        draw.text((hud_x1 + 8, 52), f"LAST: {last_str}", fill=(255, 215, 0) if last_aid is not None else (120, 130, 140), font=font)

        avail_summary = ",".join(str(i) for i in sorted(avail_set))
        draw.text((hud_x1 + 8, 68), f"AVAIL: [{avail_summary}]", fill=(180, 190, 200), font=font)

        # 下部システムボタン: RESET (0)
        rst_w, rst_h = 70, 22
        rst_x = (hud_x1 + hud_x2 - rst_w) // 2
        rst_y = 96
        rst_fill, rst_out, rst_txt = get_button_style(0)
        if 0 not in avail_set and 0 != last_aid:
            # RESET は常に能動的エスケープとして提示（赤系アクセント）
            rst_fill = (45, 25, 28)
            rst_out = (120, 50, 55)
            rst_txt = (220, 160, 165)
        draw.rectangle([rst_x, rst_y, rst_x + rst_w, rst_y + rst_h], fill=rst_fill, outline=rst_out)
        draw.text((rst_x + 10, rst_y + 4), "0:RESET", fill=rst_txt, font=font)

    # -------------------------------------------------------------
    # 3. 右側: アクションボタン群 (ACTION5, ACTION6/CLICK, ACTION7)
    # -------------------------------------------------------------
    right_x = max(hud_x2 + 20, width - 130)

    # ACTION6 (CLICK / INTERACT) - メインボタン (大きめの丸型/角丸)
    btn6_fill, btn6_out, btn6_txt = get_button_style(6)
    b6_x, b6_y, b6_r = right_x + 60, 48, 22
    draw.ellipse([b6_x - b6_r, b6_y - b6_r, b6_x + b6_r, b6_y + b6_r], fill=btn6_fill, outline=btn6_out, width=2)
    draw.text((b6_x - 14, b6_y - 10), "6:ACT", fill=btn6_txt, font=font)
    draw.text((b6_x - 16, b6_y + 2), "CLICK", fill=btn6_txt, font=font)

    # ACTION5 (上部補助ボタン)
    btn5_fill, btn5_out, btn5_txt = get_button_style(5)
    b5_x, b5_y, b5_r = right_x + 12, 34, 16
    draw.ellipse([b5_x - b5_r, b5_y - b5_r, b5_x + b5_r, b5_y + b5_r], fill=btn5_fill, outline=btn5_out)
    draw.text((b5_x - 8, b5_y - 5), "5", fill=btn5_txt, font=font)

    # ACTION7 (下部補助ボタン)
    btn7_fill, btn7_out, btn7_txt = get_button_style(7)
    b7_x, b7_y, b7_r = right_x + 18, 86, 16
    draw.ellipse([b7_x - b7_r, b7_y - b7_r, b7_x + b7_r, b7_y + b7_r], fill=btn7_fill, outline=btn7_out)
    draw.text((b7_x - 8, b7_y - 5), "7", fill=btn7_txt, font=font)

    draw.text((right_x + 20, 114), "ACTIONS", fill=(130, 145, 165), font=font)

    return panel_img


def render_console_observation(
    grid: np.ndarray,
    available_actions: Optional[List[Union[int, str]]] = None,
    last_action: Optional[Union[int, str, Any]] = None,
    step_index: int = 0,
    game_state: str = "PLAYING",
    cell_size: int = 16,
    min_console_width: int = 380,
    padding: int = 12,
    cursor_pos: Optional[Tuple[int, int]] = None,
) -> Image.Image:
    """ゲーム盤面とレトロコントローラー・HUDを一体化した統合コンソール画像を生成する.

    Args:
        grid: (H, W) のゲーム画面グリッド
        available_actions: 有効なアクションID/名前リスト
        last_action: 直前に実行されたアクション
        step_index: 現在のステップ番号
        game_state: ゲーム状態文字列
        cell_size: グリッドのセルサイズ (ピクセル)
        min_console_width: コンソール画像の最小幅
        padding: 盤面周囲の余白 (ピクセル)
        cursor_pos: (col, row) のマウスカーソル位置

    Returns:
        上部にゲーム画面、下部にコントローラーが配置された統合 PIL Image
    """
    game_img = render_grid_to_image(grid, cell_size=cell_size, cursor_pos=cursor_pos)

    # コンソール幅の決定（ゲーム画面または最小幅の大きい方）
    content_w = game_img.width
    console_w = max(min_console_width, content_w + padding * 2)

    panel_h = 135
    header_h = 24
    total_h = header_h + padding + game_img.height + padding + panel_h + padding

    # コンソール筐体ベース (ダークスレート #1A1D24)
    console_img = Image.new("RGB", (console_w, total_h), color=(18, 20, 26))
    draw = ImageDraw.Draw(console_img)
    font = ImageFont.load_default()

    # ヘッダーバー
    draw.rectangle([0, 0, console_w, header_h], fill=(28, 32, 42))
    header_title = "=== ARC-AGI-3 GAME CONSOLE ==="
    draw.text((12, 6), header_title, fill=(210, 220, 235), font=font)

    # マウスカーソル座標のヘッダー表示
    if cursor_pos is not None:
        cursor_txt = f"CURSOR: ({cursor_pos[0]}, {cursor_pos[1]})"
        draw.text((console_w - 130, 6), cursor_txt, fill=(255, 230, 80), font=font)

    # ゲーム画面の中央配置
    game_x = (console_w - game_img.width) // 2
    game_y = header_h + padding

    # ゲーム画面のベゼル/枠線
    draw.rectangle(
        [game_x - 2, game_y - 2, game_x + game_img.width + 1, game_y + game_img.height + 1],
        outline=(55, 65, 80),
        width=2,
    )
    console_img.paste(game_img, (game_x, game_y))

    # コントローラーパネルの生成と配置
    panel_w = console_w - padding * 2
    panel_img = render_gamepad_panel(
        width=panel_w,
        height=panel_h,
        available_actions=available_actions,
        last_action=last_action,
        step_index=step_index,
        game_state=game_state,
    )
    panel_y = game_y + game_img.height + padding
    console_img.paste(panel_img, (padding, panel_y))

    return console_img

