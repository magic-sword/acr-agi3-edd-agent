# Failure Diagnoser Reference Guide (ACR-AGI-3)

## Overview
Failure Diagnoser analyzes why an action policy or synthesized skill failed in gameplay, categorizing the failure mode into an abstract, environment-independent taxonomy and producing concise repair directives.

## Abstract Failure Taxonomy
1. **ContractViolation**:
   - Condition: Missing `choose_action` function, incorrect signature, or returning non-`Action` types.
   - Directive: Enforces standard interface contract `def choose_action(obs: np.ndarray, info: dict | None = None) -> Action:`.

2. **Language / Runtime Exception**:
   - Condition: `SyntaxError`, `IndentationError`, `NameError`, or NumPy boolean evaluation ambiguity (`The truth value of an array...`).
   - Directive: Generates concise syntax corrections (e.g., using `np.argwhere` instead of direct array equality, fixing indentations).

3. **BehavioralStagnation**:
   - Condition: Episode steps exceed limit without reaching goal, or agent oscillates between identical states.
   - Directive: Instructs LLM to prioritize distance-reducing or state-altering actions to break loops.

4. **SafetyInvariantBreach**:
   - Condition: Agent entered lethal traps or irreversible failure states.
   - Directive: Instructs agent to inspect neighbor cell safety before issuing directional actions.

## Integration Pattern
Failure Diagnoser prevents LLM context overload by compressing verbose stack traces and verbose error strings into single-line actionable directives for `skill-synthesizer`.
