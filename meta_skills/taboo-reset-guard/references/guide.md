# Taboo Reset Guard Reference Guide

## 1. The Active Reset Principle
In human gameplay analysis (`docs/HUMAN_ADAPTATION_ANALYSIS_REPORT.md`), solvers do not engage in infinite trial-and-error when a block is stuck in a dead-end corridor. Instead, they actively invoke the reset action to return the board to a clean state. The critical insight: the repair cost of disentangling a deadlock is higher than re-executing verified initial steps.

## 2. Persistent Taboo Memory
Whenever a reset is triggered, the state immediately preceding the deadlock is committed to a persistent taboo set, ensuring the agent prunes that branch in subsequent runs.
