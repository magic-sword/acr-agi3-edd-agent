#!/usr/bin/env python3
"""Episodic Memory Meta-Skill Implementation for ARC-AGI-3.

Transcript（全試行錯誤履歴）から Working Memory（作業記憶）への蒸留、
因果関係（Action -> ΔPixels, State Transition）の構造化保存、
および LLM からのセマンティック検索・直近要約を提供します。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Episode:
    """1ステップの因果関係記録."""
    step_index: int
    action_name: str
    action_id: int
    pixels_changed: int
    is_effective: bool
    state_before: str
    state_after: str
    levels_completed: int
    reflection: str = ""
    rule_hypothesis: str = ""
    timestamp: float = 0.0


class EpisodicMemoryManager:
    """エピソード記憶の管理・蒸留・検索エンジン."""

    def __init__(self, capacity: int = 100) -> None:
        self.capacity = capacity
        self.episodes: List[Episode] = []
        self.learned_rules: List[str] = []

    def record_step(
        self,
        step_index: int,
        action_name: str,
        action_id: int,
        pixels_changed: int,
        is_effective: bool,
        state_before: str = "NOT_FINISHED",
        state_after: str = "NOT_FINISHED",
        levels_completed: int = 0,
        reflection: str = "",
        rule_hypothesis: str = "",
    ) -> Episode:
        """ステップ結果を因果レコードとして記憶に追加."""
        if step_index < 0:
            raise ValueError("step_index must be non-negative")

        ep = Episode(
            step_index=step_index,
            action_name=action_name,
            action_id=action_id,
            pixels_changed=pixels_changed,
            is_effective=is_effective,
            state_before=state_before,
            state_after=state_after,
            levels_completed=levels_completed,
            reflection=reflection,
            rule_hypothesis=rule_hypothesis,
        )
        self.episodes.append(ep)
        if len(self.episodes) > self.capacity:
            self.episodes.pop(0)

        if rule_hypothesis and rule_hypothesis not in self.learned_rules:
            self.learned_rules.append(rule_hypothesis)

        return ep

    def get_working_memory_summary(self, max_recent: int = 5) -> str:
        """LLM の Working Memory に注入するための高密度 Markdown 要約を生成."""
        if not self.episodes:
            return "No previous steps in working memory."

        recent = self.episodes[-max_recent:]
        lines = [
            f"### Working Memory (Last {len(recent)} Steps / Total: {len(self.episodes)}):"
        ]

        for ep in recent:
            eff_str = "✅ Effective (Screen changed)" if ep.is_effective else "❌ Ineffective (No change / Wall)"
            lines.append(
                f"- Step {ep.step_index:02d}: Action `{ep.action_name}` -> ΔPixels: {ep.pixels_changed}, "
                f"Result: {eff_str}"
            )
            if ep.reflection:
                lines.append(f"  Reflection: {ep.reflection}")
            if ep.rule_hypothesis:
                lines.append(f"  Hypothesis: {ep.rule_hypothesis}")

        if self.learned_rules:
            lines.append("\n### Learned Causal Rules:")
            for i, r in enumerate(self.learned_rules, 1):
                lines.append(f"  {i}. {r}")

        return "\n".join(lines)

    def query_episodes(
        self,
        action_name: Optional[str] = None,
        effective_only: bool = False,
        keyword: Optional[str] = None,
    ) -> List[Episode]:
        """条件に合致する過去エピソードを検索."""
        results = []
        for ep in self.episodes:
            if action_name and ep.action_name.upper() != action_name.upper():
                continue
            if effective_only and not ep.is_effective:
                continue
            if keyword:
                kw_lower = keyword.lower()
                text = f"{ep.reflection} {ep.rule_hypothesis} {ep.action_name}".lower()
                if kw_lower not in text:
                    continue
            results.append(ep)
        return results

    def get_learned_rules(self) -> List[str]:
        """蓄積された因果ルール一覧を返却."""
        return list(self.learned_rules)

    def clear(self) -> None:
        """記憶をリセット."""
        self.episodes.clear()
        self.learned_rules.clear()

    def to_dict(self) -> Dict[str, Any]:
        """シリアライズ用辞書化."""
        return {
            "capacity": self.capacity,
            "episodes": [asdict(e) for e in self.episodes],
            "learned_rules": self.learned_rules,
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """辞書からの復元."""
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")
        self.capacity = data.get("capacity", 100)
        self.learned_rules = list(data.get("learned_rules", []))
        self.episodes.clear()
        for ed in data.get("episodes", []):
            try:
                self.episodes.append(Episode(**ed))
            except Exception:
                continue


def main():
    parser = argparse.ArgumentParser(description="Episodic Memory CLI Tool")
    subparsers = parser.add_subparsers(dest="command")

    # record コマンド
    rec_parser = subparsers.add_parser("record", help="Record a step episode")
    rec_parser.add_argument("--step", type=int, required=True, help="Step index")
    rec_parser.add_argument("--action", type=str, required=True, help="Action name (e.g. UP)")
    rec_parser.add_argument("--action-id", type=int, default=1, help="Action ID")
    rec_parser.add_argument("--pixels", type=int, default=0, help="Pixels changed")
    rec_parser.add_argument("--effective", action="store_true", help="Was action effective")
    rec_parser.add_argument("--reflection", type=str, default="", help="Reflection text")
    rec_parser.add_argument("--hypothesis", type=str, default="", help="Rule hypothesis")
    rec_parser.add_argument("--storage", type=str, default="memory.json", help="Storage file")

    # summary コマンド
    sum_parser = subparsers.add_parser("summary", help="Get working memory summary")
    sum_parser.add_argument("--recent", type=int, default=5, help="Number of recent steps")
    sum_parser.add_argument("--storage", type=str, default="memory.json", help="Storage file")

    # query コマンド
    q_parser = subparsers.add_parser("query", help="Query past episodes")
    q_parser.add_argument("--action", type=str, default=None, help="Filter by action name")
    q_parser.add_argument("--effective-only", action="store_true", help="Only effective steps")
    q_parser.add_argument("--keyword", type=str, default=None, help="Search keyword")
    q_parser.add_argument("--storage", type=str, default="memory.json", help="Storage file")

    args = parser.parse_args()
    mgr = EpisodicMemoryManager()

    storage_path = Path(getattr(args, "storage", "memory.json"))
    if storage_path.exists():
        try:
            with open(storage_path, "r", encoding="utf-8") as f:
                mgr.from_dict(json.load(f))
        except Exception:
            pass

    if args.command == "record":
        ep = mgr.record_step(
            step_index=args.step,
            action_name=args.action,
            action_id=args.action_id,
            pixels_changed=args.pixels,
            is_effective=args.effective,
            reflection=args.reflection,
            rule_hypothesis=args.hypothesis,
        )
        with open(storage_path, "w", encoding="utf-8") as f:
            json.dump(mgr.to_dict(), f, indent=2)
        print(f"Successfully recorded step {ep.step_index}: {ep.action_name}")

    elif args.command == "summary":
        summary = mgr.get_working_memory_summary(max_recent=args.recent)
        print(summary)

    elif args.command == "query":
        results = mgr.query_episodes(
            action_name=args.action,
            effective_only=args.effective_only,
            keyword=args.keyword,
        )
        print(f"Found {len(results)} matching episodes:")
        for ep in results:
            print(f"- Step {ep.step_index}: {ep.action_name}, ΔP={ep.pixels_changed}, Eff={ep.is_effective}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
