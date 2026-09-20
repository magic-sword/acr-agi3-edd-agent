from arc_agi import Arcade
import numpy as np
import pathlib
import cv2

arcade = Arcade(environments_dir=pathlib.Path('/workspace/data/competition/environment_files'))
env = arcade.make('ft09-0d8bbf25')
obs = env.reset()
arr = np.array(obs.frame)[0]
bg = 5

print("Valid actions:")
for v in env._game._get_valid_actions():
    print(" ", v.data)

print("\nColor components in bottom-right (x>=30, y>=30):")
for c in np.unique(arr):
    if c == bg:
        continue
    c_mask = (arr == c).astype(np.uint8)
    cnts, _ = cv2.findContours(c_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bx >= 30 and by >= 30:
            M = cv2.moments(cnt)
            cx = int(round(M["m10"] / M["m00"])) if M["m00"] != 0 else bx + bw // 2
            cy = int(round(M["m01"] / M["m00"])) if M["m00"] != 0 else by + bh // 2
            print(f"  Color {c}: center=({cx}, {cy}), bbox=({bx}, {by}, {bw}, {bh}), area={cv2.contourArea(cnt)}")
