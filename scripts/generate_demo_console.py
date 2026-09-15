"""Generate sample console observation image for verification."""

import numpy as np
from acr_agi3.dsl.renderer import render_console_observation

def main():
    grid = np.zeros((16, 16), dtype=int)
    grid[0, :] = 5
    grid[-1, :] = 5
    grid[:, 0] = 5
    grid[:, -1] = 5
    grid[4:7, 4:7] = 1  # 青ブロック
    grid[10, 10] = 2    # 赤 (プレイヤー自機)
    grid[8, 12] = 4     # 黄 (ゴール/ターゲット)

    img = render_console_observation(
        grid=grid,
        available_actions=["UP", "DOWN", "LEFT", "RIGHT", "CLICK"],
        last_action="RIGHT",
        step_index=12,
        game_state="PLAYING",
        cell_size=16,
    )
    output_path = "/workspace/logs/sample_console_demo.png"
    img.save(output_path)
    print(f"Demo image saved successfully: size={img.size}, path={output_path}")

if __name__ == "__main__":
    main()
