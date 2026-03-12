## Beyond Model-Level Performance: Evaluating and Interpreting Agentic Failures in Vision-Language Models

### Aim

1. **Benchmarking:** Evaluate recently released open-source multimodal models (e.g. Kimi‑VL, Qwen‑VL/Qwen3‑VL) on agentic, safety-sensitive, and multimodal tasks using a combination of existing benchmarks.
2. **Composite Benchmark:** Build a unified suite by recombining and extending tasks from prior benchmarks (Agent‑X, VisualWebArena, AgentHarm, SAE‑V, etc.), focusing on:
    - Multi-step agentic performance
    - Multimodal reasoning and grounding
    - Safety and robustness (including attacks)
3. **Mechanistic Analysis:** Explore failures with mechanistic tools (e.g. sparse autoencoders, CorrSteer-style steering) to understand and potentially modify model vulnerabilities.

Phases 1–3 (benchmarking, composite benchmark, attacks) are mandatory. Phases 4–5 (interpretability, interventions) are exploratory stretch goals.

---

### Research Questions

#### RQ1: Benchmarking
How do state-of-the-art open-source multimodal models perform on agentic and multimodal benchmarks as autonomous agents (multi-step), not just single-shot?

- **Models:** moonshotai/Kimi‑K2.5, Kimi‑VL‑A3B‑Thinking, Qwen/Qwen2-VL, Qwen3-VL
- **Benchmarks:** Agent‑X, VisualWebArena, VS‑Bench

#### RQ2: Composite Benchmark
Can a new composite benchmark combining tasks from multiple sources reveal systematic differences in multimodal capabilities and safety/robustness?

- **Sources:** Agent‑X, VisualWebArena, VS‑Bench, AgentHarm, SAE‑V

#### RQ3: Vulnerabilities & Attacks
What agentic vulnerabilities and safety failures emerge under normal and attack conditions? Can these be systematically captured as measurable failure categories and robustness metrics?

- **Attacks:** Text-based jailbreaks, vision-based perturbations, agent attack scenarios

#### ERQ4–6: Exploratory Mechanistic & Steering Questions
- Can sparse autoencoders identify features differentiating success/failure and robustness/vulnerability?
- Can CorrSteer-style activation steering improve or degrade performance/safety?
- Do information bottlenecks (oversquashing) emerge in multimodal-attention stacks during long agentic sequences?

---

### Weekly Update Structure
This folder is the first update for my MSc thesis proposal. Each week, I will add new folders summarizing progress, experiments, and findings. The `120326` folder is the first update, containing:
- Benchmarking experiments
- Composite benchmark construction
- Adversarial attack results
- Visual evidence and metrics

Future updates will build on this foundation, tracking progress across all phases and research questions.

---

## Weekly Results & Visual Evidence

[Click here for detailed results and plots in the `120326` folder.](./120326/README.md)

This section will be updated weekly with new folders summarizing progress, experiments, and findings. Each update will include:
- Benchmarking experiments
- Composite benchmark construction
- Adversarial attack results
- Visual evidence and metrics

---

*For the latest update, see the README and plots in the [120326 folder](./120326/README.md).*
