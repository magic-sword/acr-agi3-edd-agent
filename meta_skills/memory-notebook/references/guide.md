# Reference Guide: Long-Term Memory Notebook

This guide details the cognitive architecture, operations, and best practices for the `memory-notebook` meta-skill.

---

## 1. The Core Metaphor

| Component | Cognitive Function | CLI / Python API | Purpose |
| :--- | :--- | :--- | :--- |
| **Paper** (紙) | Structured Markdown Storage | `notebook.md`, `save()`, `load()` | Human-readable persistence across episodes & steps |
| **Pen** (ペン) | Writing & Summarizing | `write`, `summarize()` | Record facts, rules, and subgoals with 1-line summaries |
| **Eraser** (消しゴム) | Pruning & Deletion | `delete`, `clear()` | Invalidate refuted hypotheses and discard completed tasks |
| **Bookmark & TOC** (しおり・目次) | Progressive Disclosure | `toc`, `bookmark`, `read` | Retrieve overview in minimal tokens; unfold details on-demand |

---

## 2. Command Reference

### Table of Contents (`toc`)
Check existing knowledge without bloating prompt tokens:
```bash
python scripts/memory_notebook.py toc --format markdown
# Output:
# - [🔖 current_goal] **subgoals.reach_key** (Step 4) — *Collect key at (4, 12)*
# - **rules.controls** `[controls]` (Step 1) — *ACTION1 is UP*
```

### Selective Reading (`read`)
Load only the required section by its ID or bookmark:
```bash
python scripts/memory_notebook.py read --section "current_goal"
python scripts/memory_notebook.py read --section "rules.controls"
```

### Writing & Appending (`write`)
Add a new observation or update an existing topic:
```bash
# New section
python scripts/memory_notebook.py write \
  --section "rules.gates" \
  --title "Gate Mechanisms" \
  --content "Blue gate opens when the floor button at (5, 8) is pressed." \
  --summary "Floor button opens blue gate" \
  --tags "mechanics,gate" \
  --step 6

# Appending to existing section
python scripts/memory_notebook.py write \
  --section "rules.gates" \
  --content "Gate closes after 5 steps if player steps off the button." \
  --append \
  --step 11
```

### Bookmarks (`bookmark`)
Pin high-priority items (such as the current goal or active puzzle clue) for instant access:
```bash
# Set bookmark
python scripts/memory_notebook.py bookmark --name "active_lead" --section "hypotheses.secret_room"

# Remove bookmark
python scripts/memory_notebook.py bookmark --name "active_lead" --remove
```

### Erasing Obsolete Knowledge (`delete` / `clear`)
Clean up refuted hypotheses to prevent cognitive anchoring:
```bash
python scripts/memory_notebook.py delete --section "hypotheses.secret_room"
```

---

## 3. Recommended Naming Convention for Sections

Use hierarchical dot-separated slugs:
- `rules.<category>`: Game mechanics (e.g. `rules.controls`, `rules.hazards`, `rules.entities`)
- `map.<area>`: Spatial topology (e.g. `map.layout`, `map.dead_ends`, `map.landmarks`)
- `goals.<priority>`: Planning state (e.g. `goals.primary`, `subgoals.phase1`)
- `hypotheses.<topic>`: Active experiments (e.g. `hypotheses.green_orb`, `hypotheses.wall_pass`)

---

## 4. Integration into ADK Cognitive Loop

In the `plan_node` or `react_loop`:
1. Check `toc` at the beginning of an episode or when lost.
2. If an active bookmark exists (e.g. `current_goal`), retrieve it with `read --section current_goal`.
3. Plan and execute actions.
4. When a notable state transition occurs (e.g. key collected, door opened, death triggered), update memory with `write`.
