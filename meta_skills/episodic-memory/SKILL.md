---
name: episodic-memory
description: |
  Manages episodic memory distillation and working memory context for ARC-AGI-3 gameplay.
  Distills raw step transcripts into structured causal episodes (Action causes Outcome),
  provides compact working memory summaries, and enables causal rule hypothesis tracking.
  Do NOT use for grid visual rendering or direct game action dispatching.
license: MIT
allowed-tools: run_skill_script load_skill_resource
metadata:
  pattern: workflow
---

# Episodic Memory

## When to use
- Retrieve past action outcomes, step history, and causal consequences (Working Memory).
- Record and reflect on the outcome of the latest action (Effective movement vs Ineffective wall hit).
- Track and verify game mechanics hypotheses (e.g., "Red pixel is player", "Touching yellow clears level").
- Query past attempts under similar states to avoid repeated mistakes.

## When NOT to use
- Direct visual grid color rendering (use `visual-inspector`).
- Direct action dispatch or coordinate conversion (use `game-controller`).
- Dynamic skill code compilation (use `macro-skill-compiler`).

## Workflow
1. **Working Memory Retrieval (Before Planning)**:
   Extract recent step outcomes to ground LLM reasoning in verified causal reality:
   ```bash
   python scripts/episodic_memory.py summary --recent 5 --storage /tmp/memory.json
   ```
2. **Causal Query (Hypothesis Verification)**:
   Search past actions to test whether a direction was blocked or effective:
   ```bash
   python scripts/episodic_memory.py query --action UP --storage /tmp/memory.json
   ```
3. **Step Reflection & Storage (After Execution)**:
   Record the outcome of the action taken, including pixel change and causal hypothesis:
   ```bash
   python scripts/episodic_memory.py record --step 1 --action UP --pixels 2 --effective --reflection "Moved up without collision" --storage /tmp/memory.json
   ```

## Examples
- **Input**: `query --action UP`
  - **Output**: `Found 1 matching episodes: Step 0: UP, ΔP=2, Eff=True`
- **Input**: `summary --recent 3`
  - **Output**:
    ```markdown
    ### Working Memory (Last 1 Steps / Total: 1):
    - Step 00: Action `UP` -> ΔPixels: 2, Result: ✅ Effective (Screen changed)
      Reflection: Moved up without collision
    ```

## Output format
- High-density Markdown summary block suitable for LLM Working Memory injection.
- Structured JSON episode records for offline analysis.

## Anti-patterns to avoid
- Do not store raw multi-megabyte image bitmaps in memory (causes CUDA OOM).
- Do not record identical repetitive steps without consolidating into a loop detection notice.
- Do not invent hypothetical outcomes without grounding them in measured pixel diffs.

## Requirements & Prerequisites
- Python: >= 3.10
- Standard library only (`json`, `argparse`, `dataclasses`).

## Bundled Resources
### `scripts/` (Executable Tools - Zero-dependency)
- `scripts/episodic_memory.py`: Deterministic memory manager and CLI query tool.

### `references/` (On-Demand Knowledge)
- `references/guide.md`: Theoretical basis (Transcript vs Working Memory separation).
