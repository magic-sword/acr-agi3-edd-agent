"""ARC タスク向けコード実行・検証ツール (Google ADK 2.0 互換).

LLM が生成した変換プログラム (def transform(grid)) を実行し、
Train ペアに対する適合性を高速に検証します。
"""

import re
import traceback
from typing import Any, Dict, List

import numpy as np


def extract_python_code(text: str) -> str:
    """LLM レスポンスから Python コードブロックを抽出."""
    # ```python ... ``` または ``` ... ``` を抽出
    pattern = r"```(?:python)?\s*\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    # コードブロックがない場合で def transform が含まれていればそのまま
    if "def transform" in text:
        return text.strip()
    return text.strip()


def execute_and_verify_code(
    code: str,
    train_pairs: List[Dict[str, Any]],
    timeout_sec: float = 2.0,
) -> Dict[str, Any]:
    """生成された Python コードを訓練ペアに対して実行・検証する.

    Args:
        code: Python コード文字列 (def transform(grid) を含む必要がある)
        train_pairs: [{'input': 2D list/np.ndarray, 'output': 2D list/np.ndarray}, ...]
        timeout_sec: タイムアウト秒数

    Returns:
        検証結果 (is_valid, passed_count, total_count, errors)
    """
    clean_code = extract_python_code(code)

    import collections
    import math

    local_scope: Dict[str, Any] = {"np": np, "math": math, "collections": collections}
    safe_builtins = {
        "range": range,
        "len": len,
        "enumerate": enumerate,
        "zip": zip,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "int": int,
        "float": float,
        "bool": bool,
        "str": str,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "print": print,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "any": any,
        "all": all,
        "sorted": sorted,
        "reversed": reversed,
        "__import__": __import__,
    }
    global_scope: Dict[str, Any] = {
        "__builtins__": safe_builtins,
        "np": np,
        "math": math,
        "collections": collections,
    }

    try:
        exec(clean_code, global_scope, local_scope)
    except Exception as e:
        return {
            "is_valid": False,
            "error": f"Syntax/Compilation error: {type(e).__name__}: {e}",
            "passed_count": 0,
            "total_count": len(train_pairs),
        }

    transform_fn = local_scope.get("transform")
    if not callable(transform_fn):
        return {
            "is_valid": False,
            "error": "Function 'transform(grid)' was not defined in the code.",
            "passed_count": 0,
            "total_count": len(train_pairs),
        }

    passed_count = 0
    failures: List[Dict[str, Any]] = []

    for idx, pair in enumerate(train_pairs):
        inp = np.array(pair["input"], dtype=int)
        expected = np.array(pair["output"], dtype=int)

        try:
            # コピーを渡して破壊的変更を防止
            pred = transform_fn(inp.copy())
            if not isinstance(pred, np.ndarray):
                pred = np.array(pred, dtype=int)

            if pred.shape != expected.shape:
                failures.append(
                    {
                        "pair_index": idx,
                        "reason": (
                            f"Shape mismatch: predicted {pred.shape}, "
                            f"expected {expected.shape}"
                        ),
                    }
                )
            elif not np.array_equal(pred, expected):
                diff_count = int(np.sum(pred != expected))
                failures.append(
                    {
                        "pair_index": idx,
                        "reason": f"Grid values mismatch: {diff_count} cells differ.",
                    }
                )
            else:
                passed_count += 1
        except Exception as e:
            failures.append(
                {
                    "pair_index": idx,
                    "reason": f"Runtime error: {type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(limit=2),
                }
            )

    is_valid = passed_count == len(train_pairs)
    return {
        "is_valid": is_valid,
        "passed_count": passed_count,
        "total_count": len(train_pairs),
        "failures": failures if not is_valid else [],
    }


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

    global_scope: Dict[str, Any] = {
        "np": np,
        "math": math,
        "collections": collections,
        "Action": Action,
        "__builtins__": __builtins__,
    }
    local_scope: Dict[str, Any] = {}

    try:
        exec(clean_code, global_scope, local_scope)
    except Exception as e:
        return {
            "is_solved": False,
            "error": f"Syntax/Import error: {type(e).__name__}: {e}",
            "status": "error",
            "steps_taken": 0,
            "final_reward": -1.0,
        }

    policy_fn = (
        local_scope.get("choose_action")
        or local_scope.get("act")
        or global_scope.get("choose_action")
        or global_scope.get("act")
    )
    if not policy_fn:
        return {
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
                "is_solved": is_goal,
                "steps_taken": steps,
                "final_reward": total_reward,
                "status": step_res.info.get("status", "done"),
                "history": history,
            }

    return {
        "is_solved": False,
        "status": "timeout",
        "steps_taken": steps,
        "final_reward": total_reward,
        "history": history,
    }
