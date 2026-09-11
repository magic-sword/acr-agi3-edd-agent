---
name: grid-analyzer
description: |
  Analyzes ARC-AGI grid dimensions, color histograms, and geometric symmetries deterministically.
  Use when the user asks to analyze grid properties, extract color counts, or detect symmetry.
  Do NOT use for transforming grids or executing dynamic game action policies.
license: MIT
allowed-tools: run_skill_script
metadata:
  version: "1.0.0"
  author: "magic-sword"
  tier: 1
  pattern: "workflow"
---

# Grid Analyzer

## When to use
- Analyze ARC-AGI 2D grid dimensions (height, width).
- Compute color histograms and list unique colors present in the grid.
- Check geometric horizontal, vertical, and diagonal symmetry flags.

## When NOT to use
- Dynamic game environment action policy execution (use `env-observer` or game agents).
- Modifying or transforming grid pixel values (use specific transformation skills).

## Workflow
1. Input Inspection: Pass a 2D integer array (0-9 values) representing the ARC grid.
2. Deterministic Analysis: Run `scripts/analyze.py`:
   ```bash
   python scripts/analyze.py --input "[[1, 2], [2, 1]]"
   ```
3. Output Validation: Verify JSON output containing shape, colors, color_counts, and symmetry.

## Examples
- Input: `[[1, 2], [2, 1]]` → Output: `{"shape": [2, 2], "num_colors": 2, "colors": [1, 2], "symmetry": {"horizontal": false, "vertical": false, "diagonal": true}}`

## Output format
- Structured JSON with keys: `shape`, `num_colors`, `colors`, `color_counts`, `symmetry`.

## Anti-patterns to avoid
- Do not feed 1D arrays or jagged lists without validation.
- Do not attempt state transition or game affordance analysis with this static tool.

## Requirements & Prerequisites
- Python: >= 3.10
- External packages: numpy

## Bundled Resources
### `scripts/`
- `scripts/analyze.py`: Deterministic CLI script for grid analysis.

