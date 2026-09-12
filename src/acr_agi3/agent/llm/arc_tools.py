"""ACR-AGI-3 動的ゲームプレイ向け行動ポリシー実行・検証ツール (Google ADK 2.0 互換).

LLM が生成した行動ポリシー (def choose_action(obs) -> Action) を実行し、
動的ゲーム環境に対するゴール到達・制約遵守・安全性を高速検証します。
"""

import re
from typing import Any, Dict

import numpy as np


def extract_python_code(text: str) -> str:
    """LLM レスポンスから Python コードブロックを抽出 (関数の補完フォールバック付き)."""
    pattern = r"```(?:python)?\s*\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)
    raw_code = matches[-1].strip() if matches else text.strip()

    # すでに関数定義がある場合はそのまま返却
    if "def choose_action" in raw_code or "def act" in raw_code:
        return raw_code

    # 1.5Bモデルが関数ヘッダーを省略して単一式またはボディのみを出力した場合の自動ラップ
    lines = raw_code.strip().splitlines()
    if len(lines) == 1 and ("Action." in lines[0] or lines[0].strip().startswith("return")):
        stmt = lines[0].strip()
        if not stmt.startswith("return"):
            stmt = f"return {stmt}"
        return (
            "def choose_action(obs: np.ndarray, info: dict | None = None) -> Action:\n"
            f"    {stmt}"
        )

    # 複数行の処理ブロックの場合、全体を4スペースインデントしてラップ
    indented = "\n".join(f"    {line}" for line in lines)
    if "return " not in raw_code:
        indented += "\n    return Action.WAIT"
    return f"def choose_action(obs: np.ndarray, info: dict | None = None) -> Action:\n{indented}"


def execute_and_verify_game_policy(
    code: str,
    env: Any,
    max_steps: int = 50,
) -> Dict[str, Any]:
    """ゲームプレイ行動ポリシー (def choose_action(obs) -> Action) を環境で検証.

    Args:
        code: 行動ポリシーコード文字列
        env: GameEnvironment インスタンス
        max_steps: 1試行の最大許容ステップ数

    Returns:
        検証結果 (is_solved: bool, steps_taken: int, final_reward: float, error: str)
    """
    from acr_agi3.game.env import Action

    clean_code = extract_python_code(code)

    import collections
    import math

    from acr_agi3.agent.llm.edd_tools import edd_execute_game_skill, edd_list_skills

    global_scope: Dict[str, Any] = {
        "np": np,
        "math": math,
        "collections": collections,
        "Action": Action,
        "edd_execute_game_skill": edd_execute_game_skill,
        "edd_list_skills": edd_list_skills,
        "__builtins__": __builtins__,
    }

    try:
        exec(clean_code, global_scope)
    except Exception as e:
        return {
            "success": False,
            "is_solved": False,
            "error": f"Syntax/Import error: {type(e).__name__}: {e}",
            "status": "error",
            "steps_taken": 0,
            "final_reward": -1.0,
        }

    policy_fn = global_scope.get("choose_action") or global_scope.get("act")
    if not policy_fn:
        return {
            "success": False,
            "is_solved": False,
            "status": "error",
            "error": "Function 'choose_action' or 'act' not defined in policy code.",
            "steps_taken": 0,
            "final_reward": -1.0,
        }

    obs = env.reset()
    total_reward = 0.0
    steps = 0
    history = []

    for step_idx in range(1, max_steps + 1):
        steps = step_idx
        try:
            act_val = policy_fn(obs)
            if isinstance(act_val, str):
                action = Action.from_str(act_val)
            elif isinstance(act_val, int):
                action = Action(act_val)
            else:
                action = act_val
        except Exception as e:
            return {
                "success": False,
                "is_solved": False,
                "status": "error",
                "error": f"Policy execution error at step {step_idx}: {e}",
                "steps_taken": steps,
                "final_reward": total_reward,
                "history": history,
            }

        step_res = env.step(action)
        obs = step_res.observation
        total_reward += step_res.reward
        history.append(
            {
                "step": step_idx,
                "action": action.name if hasattr(action, "name") else str(action),
                "reward": step_res.reward,
                "done": step_res.done,
                "info": step_res.info,
            }
        )

        if step_res.done:
            is_goal = step_res.reward > 0 or step_res.info.get("status") == "goal_reached"
            return {
                "success": is_goal,
                "is_solved": is_goal,
                "steps_taken": steps,
                "final_reward": total_reward,
                "status": step_res.info.get("status", "done"),
                "history": history,
            }

    return {
        "success": False,
        "is_solved": False,
        "status": "timeout",
        "steps_taken": steps,
        "final_reward": total_reward,
        "history": history,
    }
