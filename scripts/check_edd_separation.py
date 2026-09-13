#!/usr/bin/env python3
"""
Pre-commit Guard: EDD Framework Separation Enforcer.

本リポジトリ (acr-agi3-edd-agent) に汎用 EDD フレームワーク本体のコードが
誤ってコミットされることを決定論的に防止するガードレール。
"""

import os
import sys
import subprocess
from pathlib import Path


def get_staged_files() -> list[str]:
    """Git のステージングされたファイル一覧を取得."""
    try:
        res = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            capture_output=True,
            text=True,
            check=True,
        )
        return [f.strip() for f in res.stdout.splitlines() if f.strip()]
    except Exception as e:
        print(f"⚠️ [EDD Guard] Failed to get staged files: {e}", file=sys.stderr)
        return []


def check_staged_files(staged_files: list[str]) -> bool:
    """ステージングファイルを検査し、違反があれば False を返す."""
    # 汎用フレームワークとして禁止されるパスパターン
    forbidden_prefixes = [
        "edd-agent-tools/",
        "src/skills/",
    ]

    # src/acr_agi3/edd/ 内のファイル変更がある場合の検査
    edd_adapter_files = [f for f in staged_files if f.startswith("src/acr_agi3/edd/")]

    violations = []
    for f in staged_files:
        for prefix in forbidden_prefixes:
            if f.startswith(prefix):
                violations.append((f, f"Must be managed in upstream repository: skill-edd-agent"))

    if violations:
        print("\n" + "=" * 78, file=sys.stderr)
        print("🚨 [EDD GUARD ERROR] 誤ったリポジトリへのコミットを検知・阻止しました！", file=sys.stderr)
        print("=" * 78, file=sys.stderr)
        print("汎用 EDD フレームワークの資産がステージングされています：", file=sys.stderr)
        for path, reason in violations:
            print(f"  ❌ {path} -> {reason}", file=sys.stderr)
        print("\n💡 正しい手順:", file=sys.stderr)
        print("  1. 上流リポジトリ /home/prog/work/skill-edd-agent で作業・コミットしてください。", file=sys.stderr)
        print("  2. 本リポジトリ (acr-agi3-edd-agent) のステージングからこれらを除外 (git reset) してください。", file=sys.stderr)
        print("=" * 78 + "\n", file=sys.stderr)
        return False

    # src/acr_agi3/edd/ が変更されている場合の注意喚起（緊急回避環境変数があればスキップ）
    if edd_adapter_files and not os.environ.get("ALLOW_EDD_ADAPTER_COMMIT"):
        # ファイル内容を検査し、独自クラス定義が巨大（汎用フレームワークの二重実装）でないか確認
        for f in edd_adapter_files:
            file_path = Path(f)
            if file_path.exists():
                text = file_path.read_text(encoding="utf-8")
                # edd_agent_tools からの委譲ではなく、巨大な独自実装がある場合を検知
                if "class DiagnosticAnalyzer" in text and "from edd_agent_tools" not in text:
                    print("\n" + "=" * 78, file=sys.stderr)
                    print(f"🚨 [EDD GUARD ERROR] {f} に汎用クラスの直接実装が検知されました！", file=sys.stderr)
                    print("=" * 78, file=sys.stderr)
                    print("汎用ロジックは edd_agent_tools からインポートするか、上流リポジトリで管理してください。", file=sys.stderr)
                    print("純粋なアダプター修正としてコミットする場合は以下を設定して再実行してください：", file=sys.stderr)
                    print("  ALLOW_EDD_ADAPTER_COMMIT=1 git commit ...", file=sys.stderr)
                    print("=" * 78 + "\n", file=sys.stderr)
                    return False

    return True


def main():
    staged = get_staged_files()
    if not staged:
        sys.exit(0)

    if not check_staged_files(staged):
        sys.exit(1)

    print("🛡️ [EDD Guard] リポジトリ境界チェック合格: 汎用EDDコードの混入なし")
    sys.exit(0)


if __name__ == "__main__":
    main()
