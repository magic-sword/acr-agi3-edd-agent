---
name: memory-notebook
description: |
  Long-term memory management engine for autonomous ARC-AGI-3 agents following the paper, pen, eraser, and bookmark metaphor.
  Provides structured markdown storage (paper), writing and summarization (pen), deletion and pruning (eraser),
  and table-of-contents index lookup with selective progressive disclosure (bookmark).
  Do NOT use for one-off primitive single-step action execution or raw pixel manipulation.
license: MIT
allowed-tools: run_skill_script load_skill_resource
metadata:
  pattern: workflow
  version: "1.1.0"
  adk_additional_tools:
    - memory_write
    - memory_read
    - memory_toc
    - memory_search
  inputs:
    - name: command
      type: str
      description: Memory operation (toc, read, write, delete, bookmark, search, clear)
    - name: section
      type: optional[str]
      description: Section identifier slug (e.g. 'rules.movement', 'hypotheses.switches') or bookmark name
    - name: content
      type: optional[str]
      description: Body text to write or append
    - name: title
      type: optional[str]
      description: Display title for the section
    - name: summary
      type: optional[str]
      description: Concise 1-2 sentence summary for TOC indexing
    - name: bookmark
      type: optional[str]
      description: Bookmark name to tag critical sections (e.g. 'mission_goal', 'key_rule')
    - name: tags
      type: optional[list[str]]
      description: Classification tags
    - name: query
      type: optional[str]
      description: Search query string
  outputs:
    - name: status
      type: str
      description: Operation execution status ('ok' or 'error')
    - name: toc
      type: optional[list[dict] | str]
      description: Table of contents index entries
    - name: section
      type: optional[dict]
      description: Retrieved section payload (Progressive Disclosure)
    - name: results
      type: optional[list[dict]]
      description: Search matches
---

# Memory Notebook

## When to use
- When recording newly discovered rules, object affordances, or causal dynamics (Pen & Paper).
- When formulating or testing multi-step subgoals and hypotheses (Pen).
- When needing an overview of existing knowledge without cluttering the context window (Bookmark & TOC).
- When retrieving detailed information for a specific topic or goal on demand (Read / Selective Disclosure).
- When invalidating disproved hypotheses, obsolete plans, or clearing finished tasks (Eraser).

## When NOT to use
- For raw pixel rendering or direct visual Gestalt classification (use `visual-inspector`).
- For executing primitive game actions (UP, DOWN, click_at) into the game environment (use `game-controller`).
- When dumping megabytes of raw sensor data without summarization.

## Workflow
1. **Overview & Index Lookup (Bookmark & TOC)**:
   - Before formulating a plan, check the table of contents to see what is already known:
     ```bash
     python scripts/memory_notebook.py toc --format markdown
     ```
   - This returns a lightweight index (section IDs, titles, summaries, tags, bookmarks) using minimal tokens.
2. **Selective Reading (Progressive Disclosure)**:
   - If a specific topic or bookmark is relevant to the current situation, read only that section:
     ```bash
     python scripts/memory_notebook.py read --section "rules.controls"
     ```
3. **Recording & Updating Knowledge (Pen)**:
   - When an action reveals new causal mechanics or confirms a hypothesis, write or append to the notebook:
     ```bash
     python scripts/memory_notebook.py write --section "rules.hazards" --title "Hazard Dynamics" --content "Red tiles trigger instant death." --tags "danger,trap" --summary "Red tiles are fatal"
     ```
   - Pin high-priority goals or rules with a bookmark:
     ```bash
     python scripts/memory_notebook.py bookmark --name "current_goal" --section "subgoals.reach_key"
     ```
4. **Pruning & Invalidation (Eraser)**:
   - When a hypothesis is disproved or a temporary subgoal is completed, erase it to keep memory clean:
     ```bash
     python scripts/memory_notebook.py delete --section "hypotheses.blue_door"
     ```

## Examples
- Example 1 (Checking known rules):
  - Agent runs `toc`, sees `[🔖 current_goal] **subgoals.reach_key** — *Collect key at (4, 12)*`.
- Example 2 (Updating learned controls):
  - Agent observes ACTION1 moved the player UP, runs `write --section "rules.controls" --content "ACTION1 moves UP 1 cell." --tags "controls"`.
- Example 3 (Erasing false premise):
  - Moving right bumped into an invisible barrier, invalidating path hypothesis → runs `delete --section "hypotheses.path_east"`.

## Output format
- Structured JSON response containing operation status (`ok` / `error`) and corresponding payloads (`toc`, `section`, `results`).
- Markdown representation available for human inspection and lightweight TOC browsing.

## Anti-patterns to avoid
- Do not dump entire conversation histories or full observation grids into a single section; summarize key insights.
- Do not read all sections into context at once; always inspect the `toc` first and selectively load only necessary sections.
- Do not leave disproved hypotheses unpruned; active pruning prevents hallucination.

## Requirements & Prerequisites
- Python: >= 3.10
- Dependencies: standard library (argparse, dataclasses, datetime, json, os, pathlib, re, sys, typing)

## Bundled Resources
### `scripts/` (Executable Tools - Zero-dependency)
- `scripts/memory_notebook.py`: Deterministic CLI tool for paper, pen, eraser, and bookmark operations.

### `references/` (On-Demand Knowledge)
- `references/guide.md`: Specifications, schema details, and cognitive integration guidelines.
