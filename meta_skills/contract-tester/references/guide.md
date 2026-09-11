# Contract Tester Reference Guide (ACR-AGI-3)

## Overview
Contract Tester is the EDD firewall gate. It guarantees that any synthesized skill achieves 100% success on its contractual specification before being deployed in gameplay.

## Firewall Gate Rules
1. **Zero-Tolerance Policy**: If any positive or negative test case fails (exception, shape mismatch, trap contact, timeout), the skill is REJECTED immediately.
2. **Deterministic Isolation**: Tests must execute with fixed seeds and isolated state to ensure reproducibility.
3. **Structured Diagnostics**: On failure, serialize exact input frame, attempted action sequence, step count, and stack trace for `failure-diagnoser`.
