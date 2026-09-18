#!/usr/bin/env python3
"""Generate high-resolution architecture diagram PNG image."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(16, 20), facecolor="#0f172a")
ax.set_facecolor("#0f172a")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# Title
ax.text(50, 97, "ARC-AGI-3 Agent Architecture & Cognitive Workflow", 
        fontsize=20, fontweight="bold", color="#38bdf8", ha="center")
ax.text(50, 94.5, "Closed-loop perception-cognition-action pipeline with Google ADK 2.0",
        fontsize=12, color="#94a3b8", ha="center")

def draw_box(x, y, w, h, title, subtitle="", bg="#1e293b", border="#3b82f6", text_color="#f8fafc"):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=1.2", 
                                  facecolor=bg, edgecolor=border, linewidth=2)
    ax.add_patch(rect)
    if subtitle:
        ax.text(x + w/2, y + h/2 + 1.2, title, fontsize=12, fontweight="bold", color=text_color, ha="center", va="center")
        ax.text(x + w/2, y + h/2 - 1.2, subtitle, fontsize=9.5, color="#cbd5e1", ha="center", va="center")
    else:
        ax.text(x + w/2, y + h/2, title, fontsize=11, fontweight="bold", color=text_color, ha="center", va="center")

def draw_arrow(x1, y1, x2, y2, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=2, mutation_scale=15))
    if label:
        ax.text((x1 + x2)/2 + 2, (y1 + y2)/2, label, fontsize=9, color="#fde047", fontweight="bold")

# Section 0: Init
draw_box(30, 88, 40, 5, "0. Game Initialization (Step 0)", "Visual Inspection Pause (visual-inspector: Target & Initial Inspection)", bg="#1e293b", border="#64748b")

# Section 1: Observation & Perception (visual-inspector)
draw_box(10, 72, 80, 11, "1. Visual Perception Input (meta_skills/visual-inspector)", 
         "• Integrated Console Screen (Official 10-Color Board + Glowing Controller HUD)\n• Pure Objective Facts: Dimensions (H x W), Colors, Delta Pixels (Δ), Available Actions\n* No hardcoded heuristics / player-goal guessing — True visual understanding",
         bg="#0c4a6e", border="#0284c7")

# Section 2: Cognitive Plan-Act Workflow (Google ADK 2.0)
container = patches.FancyBboxPatch((8, 38), 84, 29, boxstyle="round,pad=1.5", facecolor="#1e1b4b", edgecolor="#818cf8", linewidth=2.5, linestyle="--")
ax.add_patch(container)
ax.text(50, 64.5, "2. Streamlined Cognitive Workflow (Google ADK 2.0 Linear Pipeline)", fontsize=14, fontweight="bold", color="#c084fc", ha="center")

# Node 1: Perceive
draw_box(15, 54, 70, 7, "Node 1: perceive_node (Visual & Console Ingestion)",
         "Ingests visual canvas & objective facts into CognitiveState (No branching overhead)",
         bg="#312e81", border="#6366f1")

# Node 2: Plan
draw_box(15, 41, 70, 9, "Node 2: plan_node (Planner Agent - Local VLM / LLM)", 
         "• Human-like multimodal scene comprehension & intention formation\n• Fast Single-Turn Decision (10-15s)  -->  PlanProposal {action, reasoning, coordinates}\n* 0-rejection loop, 0-hallucination",
         bg="#881337", border="#f43f5e")

# Section 3: Action Execution Skill (game-controller)
draw_box(10, 20, 80, 12, "3. 1-Step Deterministic Action Skill (meta_skills/game-controller)", 
         "Node 3: act_node — Resolves action dynamics & snaps affordance coordinates\n• Action Key Resolution: Maps semantic 'UP' to physical key ID (e.g. ACTION3)\n• Geometric Affordance Auto-Snap: Snaps clicks (ACTION6) to object center-of-mass\n• Output: 100% Reliable ActionDecision {action_id, coordinates}",
         bg="#064e3b", border="#10b981")

# Section 4: Game Environment
draw_box(25, 6, 50, 8, "4. ARC-AGI-3 Game Environment", "env.step(action)  -->  Next Board Observation (Reward & State Update)", bg="#78350f", border="#f59e0b")

# Linear Arrows
draw_arrow(50, 88, 50, 83)
draw_arrow(50, 72, 50, 61)
draw_arrow(50, 54, 50, 50)
draw_arrow(50, 41, 50, 32)
draw_arrow(50, 20, 50, 14)

# Feedback arrow (from 4 back to 1)
ax.annotate("", xy=(10, 78), xytext=(25, 10),
            arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=2.5, linestyle="--",
                            connectionstyle="arc3,rad=-0.4", mutation_scale=15))
ax.text(3, 44, "Perception-Action Loop (Next Frame)", fontsize=11, color="#38bdf8", fontweight="bold", rotation=90)

plt.tight_layout()
plt.savefig("/workspace/docs/architecture_workflow.png", dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
print("Successfully generated /workspace/docs/architecture_workflow.png")
