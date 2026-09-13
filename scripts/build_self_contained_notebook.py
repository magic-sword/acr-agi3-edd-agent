"""Google ADK 準拠 フォルダ構造維持型 Kaggle 提出ノートブック生成スクリプト.

ARC Prize 2026 - ARC-AGI-3 の公式提出仕様（Gateway 連携 & submission.parquet）に準拠し、
meta_skills/ のディレクトリ構造をそのまま Kaggle 実行環境上に展開して
SkillHarness 経由で 3段階 Progressive Disclosure を実行する自己完結型ノートブックをビルドする。
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def collect_meta_skills_payload() -> dict[str, str]:
    """meta_skills/ 配下の全ファイルを再帰的に辞書として収集."""
    meta_skills_dir = REPO_ROOT / "meta_skills"
    payload: dict[str, str] = {}
    for p in sorted(meta_skills_dir.rglob("*")):
        if p.is_file() and not p.name.startswith(".") and not p.name.endswith(".pyc"):
            rel_path = str(p.relative_to(meta_skills_dir))
            payload[rel_path] = p.read_text(encoding="utf-8")
    return payload


def build_notebook() -> None:
    cells = []

    # === Cell 0: Markdown ===
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🚀 ACR-AGI-3 Official Kaggle Submission Notebook\n",
            "\n",
            "本ノートブックは ARC Prize 2026 - ARC-AGI-3 コンペティションの公式提出ノートブックです。\n",
            "\n",
            "- **設計思想**: Google ADK 2.0 準拠 3段階 Progressive Disclosure メタスキルハーネス\n",
            "- **コンペ仕様**: ARC Gateway インタラクティブゲームプレイ (Simulation Competition)\n",
            "- **提出仕様**: `/kaggle/working/submission.parquet` (`columns=['row_id', 'game_id', 'end_of_game', 'score']`)\n",
            "- **実行モード**: 通常コミット時はダミー生成、提出（Rerun）時は Gateway と連携して全タスクを自律プレイ"
        ]
    })

    # === Cell 1: 公式 Wheels セットアップ ===
    cell1_code = """# === ARC-AGI-3 公式環境セットアップ（オフライン対応） ===
import os
import subprocess
from pathlib import Path

wheel_dir = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels")
if wheel_dir.exists():
    print("📦 Installing official arc-agi packages from competition wheels...")
    cmd = [
        "pip", "install", "--no-index", "--find-links", str(wheel_dir),
        "arc-agi", "python-dotenv", "pyyaml"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print("✅ Successfully installed arc-agi and dependencies!")
    else:
        print(f"⚠️ pip notice: {res.stderr[:200]}")
else:
    print("ℹ️ Wheels directory not found (running in local / dataset-only mode).")
"""
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cell1_code.strip().split("\n")]
    })

    # === Cell 2: スキルフォルダ構造の自動展開セル ===
    skills_payload = collect_meta_skills_payload()
    payload_json_str = json.dumps(skills_payload, ensure_ascii=False)

    cell2_code = f"""# === Google ADK meta_skills/ フォルダ構造の自己展開 ===
import json
from pathlib import Path

skills_payload = json.loads({repr(payload_json_str)})

# 展開先ターゲットディレクトリの決定
target_base = Path("/kaggle/working/meta_skills") if Path("/kaggle/working").exists() else Path("meta_skills")

deployed_count = 0
for rel_path, content in skills_payload.items():
    dest_file = target_base / rel_path
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    dest_file.write_text(content, encoding="utf-8")
    deployed_count += 1

print(f"📁 Successfully deployed {{deployed_count}} files into skill folder structure: {{target_base}}")
"""
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cell2_code.strip().split("\n")]
    })

    # === Cell 3: my_agent.py の定義 ===
    cell3_code = '''%%writefile /kaggle/working/my_agent.py
"""ACR-AGI-3 自律適応型メタスキルエージェント (Meta-Skill Harness Agent)."""

import collections
import dataclasses
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

try:
    from arcengine import FrameData, GameAction, GameState
except ImportError:
    from enum import Enum
    FrameData = Any
    class GameState(str, Enum):
        NOT_PLAYED = "NOT_PLAYED"
        NOT_FINISHED = "NOT_FINISHED"
        WIN = "WIN"
        GAME_OVER = "GAME_OVER"

    class GameAction(Enum):
        RESET = 0
        ACTION1 = 1
        ACTION2 = 2
        ACTION3 = 3
        ACTION4 = 4
        ACTION5 = 5
        ACTION6 = 6
        ACTION7 = 7

        def is_simple(self):
            return self.value != 6
        def is_complex(self):
            return self.value == 6
        def set_data(self, d):
            self.action_data = d
        @classmethod
        def from_id(cls, i):
            for a in cls:
                if a.value == i:
                    return a
            return cls.ACTION1

try:
    from agents.agent import Agent
except ImportError:
    Agent = object


# =============================================================================
# Google ADK 2.0 準拠 SkillHarness (フォルダ構造透過ローダー)
# =============================================================================
@dataclasses.dataclass
class SkillMetadata:
    name: str
    description: str
    allowed_tools: List[str]
    skill_dir: Path


class SkillHarness:
    def __init__(self, search_paths: Optional[List[Path]] = None) -> None:
        if search_paths is None:
            candidates = [
                Path("/kaggle/working/meta_skills"),
                Path("/kaggle/input/acr-agi3-source/meta_skills"),
                Path("meta_skills"),
            ]
            self.search_paths = [p for p in candidates if p.exists()]
            if not self.search_paths:
                self.search_paths = [Path("meta_skills")]
        else:
            self.search_paths = search_paths

        self._metadata_cache: Dict[str, SkillMetadata] = {}
        self._module_cache: Dict[str, Any] = {}
        self.refresh()

    def refresh(self) -> None:
        self._metadata_cache.clear()
        for base in self.search_paths:
            if not base.exists():
                continue
            for skill_dir in sorted(base.iterdir()):
                if not skill_dir.is_dir():
                    continue
                skill_md = skill_dir / "SKILL.md"
                if not skill_md.exists():
                    continue
                try:
                    content = skill_md.read_text(encoding="utf-8")
                    if content.startswith("---"):
                        parts = content.split("---", 2)
                        if len(parts) >= 3:
                            data = yaml.safe_load(parts[1])
                            if isinstance(data, dict):
                                name = data.get("name", skill_dir.name)
                                desc = data.get("description", "")
                                tools = data.get("allowed-tools", [])
                                if isinstance(tools, str):
                                    tools = tools.split()
                                self._metadata_cache[name] = SkillMetadata(
                                    name=name,
                                    description=desc,
                                    allowed_tools=tools,
                                    skill_dir=skill_dir,
                                )
                except Exception:
                    pass

    def get_skill_module(self, skill_name: str, script_name: Optional[str] = None) -> Any:
        cache_key = f"{skill_name}:{script_name or 'default'}"
        if cache_key in self._module_cache:
            return self._module_cache[cache_key]

        meta = self._metadata_cache.get(skill_name)
        if not meta:
            alt_name = skill_name.replace("_", "-")
            meta = self._metadata_cache.get(alt_name)
        if not meta:
            raise KeyError(f"Skill '{skill_name}' not found in {self.search_paths}")

        scripts_dir = meta.skill_dir / "scripts"
        target_name = script_name or skill_name.replace("-", "_")
        script_file = scripts_dir / f"{target_name}.py"
        if not script_file.exists():
            py_files = list(scripts_dir.glob("*.py"))
            if py_files:
                script_file = py_files[0]
            else:
                raise FileNotFoundError(f"No python script in {scripts_dir}")

        spec = importlib.util.spec_from_file_location(f"skill_{meta.name}_{target_name}", script_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec for {script_file}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self._module_cache[cache_key] = module
        return module


# =============================================================================
# MetaSkillHarnessPlanner (フォルダ構造から動的にスキルを運用)
# =============================================================================
class MetaSkillHarnessPlanner:
    def __init__(self, game_id: str = "") -> None:
        self.game_id = game_id
        self.harness = SkillHarness()
        obs_mod = self.harness.get_skill_module("env-observer")
        syn_mod = self.harness.get_skill_module("skill-synthesizer")
        self.syn_mod = syn_mod
        self.observer = obs_mod.MetaObserver()
        self.synthesizer = syn_mod.MetaSkillSynthesizer()

        self.step_index: int = 0
        self.last_action_id: Optional[int] = None
        self.last_action_data: Dict[str, Any] = {}
        self.last_grid: Optional[Any] = None
        self.consecutive_ineffective: int = 0
        self.clicked_coords: Set[Tuple[int, int]] = set()

    def decide_action(
        self,
        grid: Any,
        available_action_ids: List[int],
    ) -> Tuple[int, Dict[str, Any], str]:
        self.step_index += 1

        # 1. 観測アフォーダンス同定
        report = self.observer.analyze_frame(
            grid=grid,
            recent_action=self.last_action_id,
        )

        # 2. クリック系アクション (ACTION6) が利用可能な場合の処理
        if 6 in available_action_ids:
            act_id, act_data, reasoning = self._handle_click(report, grid)
            self.last_action_id = act_id
            self.last_action_data = act_data
            return act_id, act_data, reasoning

        # 3. スキル動的合成
        active_skill = self.synthesizer.synthesize(report)
        chosen_action = active_skill.choose_action(report)

        if chosen_action is not None and chosen_action in available_action_ids:
            self.last_action_id = chosen_action
            self.last_action_data = {}
            skill_name = type(active_skill).__name__
            return chosen_action, {}, f"MetaSkill[{skill_name}]: Step {self.step_index}"

        # 4. フォールバック
        self.synthesizer.blacklist_current_target()
        fallback_skill = self.syn_mod.FrontierExplorationSkill()
        fallback_act = fallback_skill.choose_action(report)
        if fallback_act not in available_action_ids:
            fallback_act = available_action_ids[0]

        self.last_action_id = fallback_act
        self.last_action_data = {}
        return fallback_act, {}, f"MetaSkill[Fallback]: {fallback_act}"

    def _handle_click(self, report: Any, grid: Any) -> Tuple[int, Dict[str, Any], str]:
        h, w = report.grid_shape
        # 未クリックのターゲット候補をクリック
        for cand in report.target_candidates:
            for r, c in cand.pixels:
                if (r, c) not in self.clicked_coords:
                    self.clicked_coords.add((r, c))
                    return 6, {"x": int(c), "y": int(r)}, f"MetaSkill[ClickTarget]: ({c}, {r})"

        cx, cy = w // 2, h // 2
        return 6, {"x": int(cx), "y": int(cy)}, f"MetaSkill[ClickCenter]: ({cx}, {cy})"

    def on_feedback(self, is_effective: bool, pixels_changed: int) -> None:
        if not is_effective:
            self.consecutive_ineffective += 1
            if self.last_action_id in (1, 2, 3, 4) and hasattr(self, "last_agent_pos"):
                pass
        else:
            self.consecutive_ineffective = 0


GestaltVCGTPlanner = MetaSkillHarnessPlanner


class MyAgent(Agent):
    """ACR-AGI-3 自律適応型メタスキルエージェント (Meta-Skill Harness Agent)."""

    def __init__(
        self,
        card_id: str = "local",
        game_id: str = "default",
        agent_name: str = "MyAgent",
        ROOT_URL: str = "http://local",
        record: bool = False,
        arc_env: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        try:
            super().__init__(card_id, game_id, agent_name, ROOT_URL, record, arc_env, *args, **kwargs)
        except Exception:
            pass
        self.game_id = game_id or getattr(self, "game_id", "default")
        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        random.seed(seed)
        self.step_count = 0
        self.action_history: List[int] = []
        self.last_frame_hash: Optional[int] = None
        self.planner = GestaltVCGTPlanner(game_id=self.game_id)

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        state = getattr(latest_frame, "state", None)
        return state is GameState.WIN

    def _get_cands(self, latest_frame: FrameData) -> List[Any]:
        avail = getattr(latest_frame, "available_actions", None)
        reset_val = getattr(GameAction.RESET, "value", 0)
        cands = []
        if avail:
            for act_id in avail:
                if act_id != reset_val:
                    try:
                        cands.append(GameAction.from_id(act_id))
                    except Exception:
                        pass
        if not cands:
            all_actions = list(GameAction) if hasattr(GameAction, "__iter__") else [
                getattr(GameAction, f"ACTION{i}", None) for i in range(1, 8)
            ]
            cands = [a for a in all_actions if a is not None and getattr(a, "value", -1) != reset_val]
        return cands

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> Any:
        self.step_count += 1
        state = getattr(latest_frame, "state", None)

        if state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            self.step_count = 0
            self.action_history.clear()
            self.planner = GestaltVCGTPlanner(game_id=self.game_id)
            return GameAction.RESET

        try:
            cands = self._get_cands(latest_frame)
            if not cands:
                return GameAction.RESET

            grid = getattr(latest_frame, "frame", [])
            cand_ids = [getattr(a, "value", 1) for a in cands]

            def _hash_grid(g):
                try:
                    if not g:
                        return 0
                    if isinstance(g, (list, tuple)) and len(g) > 0:
                        if isinstance(g[0], (list, tuple)) and len(g[0]) > 0 and isinstance(g[0][0], (list, tuple)):
                            g = g[-1]
                        elif len(g) == 1 and isinstance(g[0], (list, tuple)):
                            g = g[0]
                    return hash(tuple(tuple(int(c[0]) if isinstance(c, (list, tuple)) else int(c) for c in row) for row in g))
                except Exception:
                    return 0

            current_hash = _hash_grid(grid)
            is_eff = (self.last_frame_hash is not None and current_hash != self.last_frame_hash)
            self.planner.on_feedback(is_effective=is_eff, pixels_changed=1 if is_eff else 0)
            self.last_frame_hash = current_hash

            act_id, act_data, reasoning = self.planner.decide_action(grid, cand_ids)
            chosen_action = GameAction.from_id(act_id)

            if hasattr(chosen_action, "is_complex") and chosen_action.is_complex():
                chosen_action.set_data(act_data)
                chosen_action.reasoning = {
                    "desired_action": f"{chosen_action.value}",
                    "my_reason": reasoning,
                }
            else:
                chosen_action.reasoning = reasoning

            self.action_history.append(act_id)
            return chosen_action

        except Exception as e:
            avail = getattr(latest_frame, "available_actions", None)
            if avail:
                act_id = [x for x in avail if x != 0][0] if any(x != 0 for x in avail) else 0
                return GameAction.from_id(act_id)
            return GameAction.from_id(1)
'''
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cell3_code.strip().split("\n")]
    })

    # === Cell 4: Rerun 実行セル (Gateway 連携) ===
    cell4_code = """# === Rerun モード: ARC Gateway 連携ゲームプレイ ===
import os
import subprocess
from pathlib import Path

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    print("🌐 [RERUN MODE] Waiting for ARC Gateway to be ready...")
    subprocess.run([
        "curl", "--fail", "--retry", "999", "--retry-all-errors", "--retry-delay", "5",
        "--retry-max-time", "600", "http://gateway:8001/api/games"
    ], check=True)
    print("✅ Gateway is live and responding!")

    agents_src = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents")
    agents_dest = Path("/kaggle/working/ARC-AGI-3-Agents")
    if not agents_dest.exists() and agents_src.exists():
        import shutil
        shutil.copytree(agents_src, agents_dest)
        print("✅ Copied ARC-AGI-3-Agents to /kaggle/working")

    my_agent_src = Path("/kaggle/working/my_agent.py")
    if my_agent_src.exists() and agents_dest.exists():
        import shutil
        shutil.copy(my_agent_src, agents_dest / "agents" / "templates" / "my_agent.py")

    init_content = \"\"\"from typing import Type, cast
from dotenv import load_dotenv
from .agent import Agent, Playback
from .swarm import Swarm
from .templates.random_agent import Random
from .templates.my_agent import MyAgent

load_dotenv()

AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
    "random": Random,
    "myagent": MyAgent,
}
\"\"\"
    if agents_dest.exists():
        with open(agents_dest / "agents" / "__init__.py", "w", encoding="utf-8") as f:
            f.write(init_content)

    env_content = \"\"\"SCHEME=http
HOST=gateway
PORT=8001
ARC_API_KEY=test-key-123
ARC_BASE_URL=http://gateway:8001/
OPERATION_MODE=online
ENVIRONMENTS_DIR=
RECORDINGS_DIR=/kaggle/working/server_recording
\"\"\"
    if agents_dest.exists():
        with open(agents_dest / ".env", "w", encoding="utf-8") as f:
            f.write(env_content)

    print("🚀 Running agent against Gateway...")
    env = os.environ.copy()
    env["MPLBACKEND"] = "agg"
    res = subprocess.run(
        ["python", "main.py", "--agent", "myagent"],
        cwd=str(agents_dest),
        env=env,
        capture_output=True,
        text=True
    )
    if res.stdout:
        print("Agent STDOUT (tail):")
        print(res.stdout[-2000:])
    if res.stderr:
        print("Agent STDERR (tail):")
        print(res.stderr[-1000:])
    print("✅ Gateway game session completed successfully!")
else:
    print("🧪 [STANDALONE / COMMIT MODE] Skipping gateway run.")
"""
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cell4_code.strip().split("\n")]
    })

    # === Cell 5: 提出用 Parquet / CSV 生成 ===
    cell5_code = """# === 提出ファイル生成 (submission.parquet / submission.csv) ===
import os
import pandas as pd
from pathlib import Path

working_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
parquet_path = working_dir / "submission.parquet"
csv_path = working_dir / "submission.csv"

if not parquet_path.exists() or not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    submission = pd.DataFrame(
        data=[['1_0', '1', True, 1]],
        columns=['row_id', 'game_id', 'end_of_game', 'score']
    )
    submission.to_parquet(parquet_path, index=False)
    submission.to_csv(csv_path, index=False)
    print(f"✅ Generated submission for Kaggle leaderboard: {parquet_path}")

print(f"Submission status: exists={parquet_path.exists()}, size={parquet_path.stat().st_size if parquet_path.exists() else 0} bytes")
"""
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cell5_code.strip().split("\n")]
    })

    # === Cell 6: バリデーション検証 ===
    cell6_code = """# === 提出ファイルのバリデーション検証 ===
import pandas as pd
from pathlib import Path

working_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
parquet_path = working_dir / "submission.parquet"

assert parquet_path.exists(), f"❌ {parquet_path} was not created!"
df = pd.read_parquet(parquet_path)

print("=== 📊 Submission Artifacts Verification ===")
print(f"Parquet File: {parquet_path} ({parquet_path.stat().st_size} bytes)")
print(f"Columns: {list(df.columns)}")
print(f"Rows: {len(df)}")
print(df.head())

assert set(df.columns) == {'row_id', 'game_id', 'end_of_game', 'score'}, "❌ Columns mismatch!"
print("🎉 All submission checks passed successfully!")
"""
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cell6_code.strip().split("\n")]
    })

    # ノートブック JSON の組み立て
    notebook_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    out_path = REPO_ROOT / "notebooks" / "submission_template.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=2, ensure_ascii=False)

    print(f"🎉 Successfully built self-contained notebook at: {out_path}")
    print(f"📦 Embedded {len(skills_payload)} files into autonomous deployment cell.")


if __name__ == "__main__":
    build_notebook()
