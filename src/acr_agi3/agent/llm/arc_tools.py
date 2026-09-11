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

    # 実行スコープの準備
    local_scope: Dict[str, Any] = {"np": np}
    global_scope: Dict[str, Any] = {
        "__builtins__": {
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
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "print": print,
            "isinstance": isinstance,
        },
        "np": np,
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
                failures.append({
                    "pair_index": idx,
                    "reason": f"Shape mismatch: predicted {pred.shape}, expected {expected.shape}",
                })
            elif not np.array_equal(pred, expected):
                diff_count = int(np.sum(pred != expected))
                failures.append({
                    "pair_index": idx,
                    "reason": f"Grid values mismatch: {diff_count} cells differ.",
                })
            else:
                passed_count += 1
        except Exception as e:
            failures.append({
                "pair_index": idx,
                "reason": f"Runtime error: {type(e).__name__}: {e}",
                "traceback": traceback.format_exc(limit=2),
            })

    is_valid = passed_count == len(train_pairs)
    return {
        "is_valid": is_valid,
        "passed_count": passed_count,
        "total_count": len(train_pairs),
        "failures": failures if not is_valid else [],
    }
