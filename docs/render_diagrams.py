#!/usr/bin/env python3
"""Generate high-resolution architecture diagram PNG image for ARC-AGI-3 Agent.

Visualizes the updated closed-loop perception-cognition-action pipeline
featuring Google ADK 2.0 Native ReAct Tool-Use Loop & ObservationTools.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(16, 22), facecolor="#090d16")
ax.set_facecolor("#090d16")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# 1. Header & Title
ax.text(50, 97.8, "ARC-AGI-3 Agent Architecture & ReAct Cognitive Workflow", 
        fontsize=21, fontweight="bold", color="#38bdf8", ha="center")
ax.text(50, 95.8, "Google ADK 2.0 Native Closed-Loop with On-Demand Observation Tools & Progressive Disclosure",
        fontsize=12, color="#94a3b8", ha="center")

def draw_box(x, y, w, h, title, subtitle="", bg="#1e293b", border="#3b82f6", text_color="#f8fafc", title_size=11, sub_size=9):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=1.2", 
                                  facecolor=bg, edgecolor=border, linewidth=2)
    ax.add_patch(rect)
    if subtitle:
        ax.text(x + w/2, y + h/2 + 1.2, title, fontsize=title_size, fontweight="bold", color=text_color, ha="center", va="center")
        ax.text(x + w/2, y + h/2 - 1.2, subtitle, fontsize=sub_size, color="#cbd5e1", ha="center", va="center")
    else:
        ax.text(x + w/2, y + h/2, title, fontsize=title_size, fontweight="bold", color=text_color, ha="center", va="center")

def draw_arrow(x1, y1, x2, y2, color="#38bdf8", label="", lw=2, linestyle="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw, linestyle=linestyle, mutation_scale=15))
    if label:
        ax.text((x1 + x2)/2 + 2, (y1 + y2)/2, label, fontsize=9.5, color="#fde047", fontweight="bold")

# Section 1: Observation & Perception (meta_skills/visual-inspector & ObservationTools)
container_s1 = patches.FancyBboxPatch((8, 77.5), 84, 16.5, boxstyle="round,pad=1.2", facecolor="#0c2d48", edgecolor="#0284c7", linewidth=2.2)
ax.add_patch(container_s1)
ax.text(50, 92.5, "1. Visual Perception & Observation Tools (meta_skills/visual-inspector & ObservationTools)", 
        fontsize=13, fontweight="bold", color="#38bdf8", ha="center")

# S1 Left: Visual Ingestion
draw_box(10, 78.8, 38, 12, "Visual Perception Ingestion", 
         "• Integrated Console Screen (10-Color Board + Glowing HUD)\n• Objective Facts: Dimensions, Colors, Δ Pixels, Action Keys\n• Pure objective visual groundings (No heuristic guessing)",
         bg="#0f3c5c", border="#38bdf8", text_color="#f0f9ff", title_size=11, sub_size=8.5)

# S1 Right: Observation Tools (ADK 2.0 FunctionTools)
draw_box(52, 78.8, 38, 12, "Google ADK 2.0 ObservationTools (FunctionTools)", 
         "• inspect_board: Geometry, color palette, style, gestalt invariants\n• inspect_affordances: Candidate player, targets, obstacles\n• inspect_action_effect: Last action Δ pixels & verified dynamics\n• inspect_roi: Close-up zoom into critical maze/pattern subgrids",
         bg="#0f4a44", border="#2dd4bf", text_color="#ccfbf1", title_size=11, sub_size=8.5)


# Section 2: Cognitive Plan-Act Workflow (Google ADK 2.0)
container_s2 = patches.FancyBboxPatch((8, 34), 84, 40, boxstyle="round,pad=1.5", facecolor="#18133b", edgecolor="#818cf8", linewidth=2.5, linestyle="--")
ax.add_patch(container_s2)
ax.text(50, 72.2, "2. Google ADK 2.0 Cognitive Workflow (ReAct Self-Iterative Planning Pipeline)", 
        fontsize=14, fontweight="bold", color="#c084fc", ha="center")

# Node 1: Perceive
draw_box(15, 64.5, 70, 6, "Node 1: perceive_node (Visual & State Ingestion)",
         "Loads integrated console image & objective context into CognitiveState",
         bg="#2e276b", border="#6366f1", title_size=11, sub_size=9)

# Node 2: Plan (Outer Box)
container_plan = patches.FancyBboxPatch((12, 35.5), 76, 26, boxstyle="round,pad=1.2", facecolor="#350e24", edgecolor="#f43f5e", linewidth=2)
ax.add_patch(container_plan)
ax.text(50, 59.8, "Node 2: plan_node (Planner Agent — Multi-Turn ReAct Reasoning Loop)", fontsize=12.5, fontweight="bold", color="#fda4af", ha="center")

# Sub-boxes inside plan_node
draw_box(14, 45, 34, 13, "Planner Agent (LLM / VLM)", 
         "1. Situation Analysis (Hypothesis)\n2. Epistemic Inquiry (Need more info?)\n3. Subgoal Formation (Goal)\n4. Action Selection (Optimal Move)\n5. Final Output: PlanProposal JSON",
         bg="#500724", border="#fb7185", title_size=10.5, sub_size=8.5)

draw_box(52, 45, 34, 13, "On-Demand Tool-Use Loop (ReAct)", 
         "Self-Directed Iterative Tool Calling:\n  ToolCall: inspect_affordances / ROI\n  ADK Runner: Executes & Feeds Result\n  Rethink: Synthesize tool feedback\n* SkillToolset: Progressive Disclosure L1/L2/L3",
         bg="#3b0764", border="#c084fc", title_size=10.5, sub_size=8.5)

# Bi-directional arrow inside ReAct Loop
ax.annotate("", xy=(52, 53), xytext=(48, 53),
            arrowprops=dict(arrowstyle="->", color="#fde047", lw=2.2, mutation_scale=14))
ax.annotate("", xy=(48, 49), xytext=(52, 49),
            arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=2.2, mutation_scale=14))
ax.text(50, 54.2, "Tool Call", fontsize=8.5, color="#fde047", fontweight="bold", ha="center")
ax.text(50, 47.5, "Response", fontsize=8.5, color="#38bdf8", fontweight="bold", ha="center")

# Output of Node 2
draw_box(20, 36.5, 60, 6.5, "Final Decision Output: PlanProposal", 
         "{\n  \"hypothesis\": \"...\", \"goal\": \"...\", \"action\": \"UP\", \"reasoning\": \"...\", \"coords\": {...}\n}",
         bg="#4c0519", border="#f43f5e", text_color="#fecdd3", title_size=10, sub_size=8)


# Section 3: Action Execution Skill (meta_skills/game-controller)
draw_box(10, 18, 80, 12, "3. 1-Step Deterministic Action Skill (meta_skills/game-controller)", 
         "Node 3: act_node — Dynamic keybinding resolution & affordance auto-snap\n• Action Key Resolution: Maps semantic 'UP' to physical key ID (e.g. ACTION3)\n• Geometric Affordance Auto-Snap: Snaps click coordinates (ACTION6) to object center-of-mass\n• Output: 100% Reliable ActionDecision {action_id, coordinates}",
         bg="#064e3b", border="#10b981", title_size=12, sub_size=9)

# Section 4: Game Environment
draw_box(25, 4.5, 50, 8.5, "4. ARC-AGI-3 Interactive Game Environment", 
         "env.step(action)  -->  Next Board Observation (Reward, Feedback & State Update)", 
         bg="#78350f", border="#f59e0b", title_size=11.5, sub_size=9)


# Arrows between main sections
draw_arrow(50, 77.5, 50, 70.8)
draw_arrow(50, 64.5, 50, 61.8)
draw_arrow(50, 36.5, 50, 30.2)
draw_arrow(50, 18, 50, 13.2)

# Global Feedback Loop (from Env back to Perception)
ax.annotate("", xy=(8, 86), xytext=(25, 9),
            arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=2.5, linestyle="--",
                            connectionstyle="arc3,rad=-0.4", mutation_scale=15))
ax.text(2.5, 48, "Perception-Action Closed Loop (Next Frame)", fontsize=11, color="#38bdf8", fontweight="bold", rotation=90)

plt.tight_layout()
out_path = "/workspace/docs/architecture_workflow.png"
plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
print(f"Successfully generated {out_path}")
