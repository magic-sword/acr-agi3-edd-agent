# Visual Inspector Reference Guide

## 1. Visual Inspection Pause Protocol
Human gameplay analysis (`docs/HUMAN_ADAPTATION_ANALYSIS_REPORT.md`) demonstrates that top human solvers consistently pause during the first 0–6 seconds upon entering a new puzzle layout. Premature action in interactive physics environments often triggers irreversible state collapse (e.g. dropping a block that cannot be lifted, or blocking a narrow corridor).

## 2. Gestalt Difference Taxonomy
1. **IDENTITY**: Current layout matches target. Action: Trigger clearance or advance.
2. **TRANSLATION_ONLY**: Relative order matches, but coordinate offset exists. Action: Direct linear push or pull.
3. **REVERSAL_REORDER**: Elements exist in opposite or permuted order (e.g. green-blue-orange-red to red-orange-blue-green).
   * Invariant: Requires intermediate separation and buffer staging area.
4. **INTERLEAVED_ALTERNATING**: Target requires interleaving different colors (e.g. red-blue-red).
   * Invariant: Requires multi-lane bypass routing (upper and lower routes).
5. **CROSS_INTERSECTION**: Target requires intersecting perpendicular lines sharing a common center piece.
   * Invariant: Identify the anchor piece (shared center) and fix it first.
