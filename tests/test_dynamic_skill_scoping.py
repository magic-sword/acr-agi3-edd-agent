"""動的スキルスコープ (Dynamic Skill Scoping) のドメイン独立トリガー・隔離テスト."""

import numpy as np

from acr_agi3.meta.intuitor import GameStyleIntuitor


def test_dynamic_skill_scoping_domain_isolation():
    """環境切り替え時にスキルがドメインごとに独立してトリガー・隔離されるかを検証."""
    intuitor = GameStyleIntuitor()

    # ドメイン別スキルプール (具象スキルのストレージ)
    domain_skills_pool = {
        "navigation": [],
        "inventory_puzzle": [],
        "exploration": [],
        "hazard_avoidance": [],
    }

    # 1. ナビゲーション迷路でスキルを獲得し、navigation ドメインに格納
    maze_skill = {
        "name": "skill_nav_maze_obstacle_bypass",
        "desc": "Check adjacent walls and detour around obstacle column toward goal.",
        "code": "def choose_action(obs, info): ...",
    }
    domain_skills_pool["navigation"].append(maze_skill)

    # 2. Case A: 同一ドメイン (navigation 迷路) の盤面観測
    obs_maze = np.zeros((8, 8), dtype=int)
    obs_maze[0, :] = 1
    obs_maze[7, :] = 1
    obs_maze[:, 0] = 1
    obs_maze[:, 7] = 1
    obs_maze[2:6, 3] = 1
    obs_maze[1, 1] = 2
    obs_maze[6, 6] = 3

    res_maze = intuitor.analyze_style(obs_maze)
    domain_maze = res_maze["recommended_domain"]
    active_skills_maze = domain_skills_pool.get(domain_maze, [])

    # 検証 A: 同一環境では保存された迷路スキルが正しくトリガーされる
    assert domain_maze == "navigation"
    assert len(active_skills_maze) == 1
    assert active_skills_maze[0]["name"] == "skill_nav_maze_obstacle_bypass"

    # 3. Case B: 異なるドメイン (inventory_puzzle: 鍵と扉がある環境)
    obs_puzzle = np.zeros((8, 8), dtype=int)
    obs_puzzle[0, :] = 1
    obs_puzzle[7, :] = 1
    obs_puzzle[:, 0] = 1
    obs_puzzle[:, 7] = 1
    obs_puzzle[1, 1] = 2  # プレイヤー
    obs_puzzle[6, 6] = 3  # ゴール
    obs_puzzle[2, 5] = 4  # 鍵 (孤立アイテム)

    res_puzzle = intuitor.analyze_style(obs_puzzle)
    domain_puzzle = res_puzzle["recommended_domain"]
    active_skills_puzzle = domain_skills_pool.get(domain_puzzle, [])

    # 検証 B: 異環境では前環境の迷路スキルが物理的に遮断され、トリガーされない (隔離)
    assert domain_puzzle == "inventory_puzzle"
    assert len(active_skills_puzzle) == 0  # 迷路スキルは混入しない

    # 4. Case C: 異なるドメイン (hazard_avoidance: 致死溶岩帯がある環境)
    obs_hazard = np.zeros((8, 8), dtype=int)
    obs_hazard[0, :] = 1
    obs_hazard[7, :] = 1
    obs_hazard[:, 0] = 1
    obs_hazard[:, 7] = 1
    obs_hazard[1, 1] = 2
    obs_hazard[6, 6] = 3
    obs_hazard[2:6, 4] = 4  # 4マスの溶岩帯

    res_hazard = intuitor.analyze_style(obs_hazard)
    domain_hazard = res_hazard["recommended_domain"]
    active_skills_hazard = domain_skills_pool.get(domain_hazard, [])

    # 検証 C: 溶岩環境でも前環境の迷路スキルは完全に遮断される
    assert domain_hazard == "hazard_avoidance"
    assert len(active_skills_hazard) == 0
