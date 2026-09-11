# 🚀 ARC-AGI-3: Human VCGT Dataset (Video &amp; LLM-Annotated Explanations)

Welcome! The **Human Trajectory Video-Cognition Goal-Task (VCGT) Dataset** is designed to provide a solid foundation for solving the complex tasks in ARC-AGI-3 using a top-down, **imitation learning** approach.

By reconstructing raw human execution logs into visual gameplay videos and mapping them to structured, hierarchical natural language explanations via an advanced LLM, this dataset bridges the gap between raw integer trajectories and high-level human reasoning.

---

## 🎯 Motivation &amp; Core Concept

Solving ARC-AGI-3 puzzles requires long-term planning and intricate operational procedures. In such environments, bottom-up methods (like unsupervised reinforcement learning based on pure exploration) suffer heavily from sparse rewards and massive search spaces. 

Instead, a **top-down approach**—learning directly from how human experts think and act—presents a much more viable path.In fact, training on natural language reasoning (like the  [H-ARC](https://github.com/Le-Gris/h-arc) dataset) was a proven key factor for the **ARC Prize 2025 winner's solution [(NV-ARC)](https://www.kaggle.com/competitions/arc-prize-2025/writeups/nvarc)**.

### The Challenge with the Raw Human Dataset
The official [ARC-AGI-3 Human Dataset](https://arcprize.org/blog/arc-agi-3-human-dataset) released by the ARC Foundation is a fantastic resource, but it was collected primarily for statistical validation. 
* It contains **no natural language explanations** of the human thought process.
* The raw logs consist purely of integer frames and button input symbols, making it incredibly difficult for an AI (or even a human) to intuitively capture the underlying strategy.

To solve this, this dataset infuses the raw play logs with both **visual reconstruction** and **dense reasoning annotations**.

---

## 🛠️ How it was Built (Methodology)

### 1. Replaying the Game in Video Format
Human players interact with the environment through a console UI (as seen on the ARC-AGI tasks page). Visual cues like colored grids, D-pad layouts, and the exact coordinates/timing of mouse clicks are vital for understanding the intent. 
* We parsed the raw frame data and reconstructed them into **full gameplay videos**, explicitly visualizing click coordinates and input timing with a custom cursor icon to make the trajectories easily digestible for multimodal models.

🎬 **Open-Source Notebook:**
The script used to generate these gameplay videos from the raw data has been fully open-sourced here:
👉 [ARC-AGI-3: Generate Gameplay Videos (Kaggle Notebook)](https://www.kaggle.com/code/magicsword001/generate-gameplay-videos?scriptVersionId=331155112)

### 2. Domain-Specific Pre-Exploratory Context
Through testing, we found that letting LLMs blindly analyze the videos led to poor results, especially in low-motion sequences (e.g., the subtle extension/retraction of a piston bar in the `sk48` task)[cite: 11, 12]. 
* To overcome this, we provided the LLM with a clear, domain-specific rulebook using intuitive analogies (e.g., *"extending the piston will impale the block"*)[cite: 12, 13]. This drastically improved tracking accuracy, ensuring the LLM correctly identified critical, minute operations.
* 🔗 *Rulebook Dataset:* [ARC-AGI-3-Game-Rules](https://www.kaggle.com/datasets/magicsword001/arc-agi-3-game-rules) 

### 3. Strict Hierarchical Output Structuring
To prevent generic summaries, the LLM was constrained to output annotations using a strict, deep hierarchical format:
* **Goal Timestamp &amp; Objective:** What is the player trying to achieve? [cite: 14, 21]
* **Reasoning Breakdown:** Why? (Recursively broken down until intuitively obvious) 
* **Task Decomposition:** What steps are needed? (Broken down into atomic actions) 
* **Action Logs:** The actual execution inputs.
* *This block repeats every time the player's immediate goal shifts.*

### 4. LLM Choice
Annotations were generated using **ChatGPT-5.5 (high model)**, which demonstrated the most logically sound, structurally stable, and precise analytical text compared to other models like Gemini-3.5 Flash (Thinking) and Gemini-3.1 Pro.

---

## 📂 Dataset Structure &amp; Contents

This dataset contains both tabular logs with deep LLM reasoning annotations and reconstructed visual files. Below is the exact file structure and data schemas for this dataset.

### 1. Tabular Trajectory &amp; Analysis Data (`results.csv`)
The primary data is stored in `results.csv`. It maps the raw ARC-AGI-3 human play trajectories directly to the generated LLM reasoning and the generated video file paths.

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `env` | `string` | The environment name / task environment (e.g., `sk48`). |
| `guid` | `string` | Unique identifier for a specific user session or trajectory instance. |
| `trajector...