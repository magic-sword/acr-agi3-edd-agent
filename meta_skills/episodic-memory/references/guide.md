# Episodic Memory Guide & Architecture

## Transcript vs Working Memory Separation
In complex interactive game playing, raw experience streams (Transcripts) grow unbounded over dozens of steps.
Storing full multi-modal history in LLM context causes linear VRAM expansion and CUDA Out-Of-Memory (OOM).

Episodic Memory separates raw historical storage from the working context:
1. **Raw Transcript (External Storage)**: Detailed telemetry, frames, and full dialogue.
2. **Distilled Episodes (Structured Records)**: Step index, action taken, pixel delta, effectiveness, and causal rule.
3. **Working Memory (In-Context)**: High-density Markdown summary injected dynamically into the prompt.
