# 🧠 AdaThink-Med

> **AdaThink-Med: Medical Adaptive Thinking with Uncertainty-Guided Length Calibration**  
> *End-to-end adaptive reasoning framework for efficient medical large language models.*

---

## 🚀 Overview

Recent progress in **medical reasoning large language models (LLMs)** has led to impressive accuracy but also excessive *overthinking* — producing long, redundant reasoning chains even for simple problems.  
**AdaThink-Med** introduces an **uncertainty-guided adaptive thinking mechanism**, allowing the model to dynamically decide **when to think more and when to think less.**

### ✨ Key Features
- 🧩 **Uncertainty-Guided Length Calibration** — adapt reasoning depth to problem difficulty.  
- ⚖️ **Performance–Efficiency Trade-off** — reduce inference cost up to **6.4×** with minimal accuracy drop.  
- 🔄 **Staged Adaptive Training** — ensures stable reinforcement learning and avoids reward hacking.  
- 🧠 **Emergent Dual Modes** — model autonomously exhibits “thinking” and “non-thinking” behaviors.  
- 📊 **Validated on Six Public Medical QA Benchmarks** with **state-of-the-art accuracy-efficiency score (AES)**.

---

## 🧬 Resources

| Resource | Link |
|-----------|------|
| 📄 **Paper** | [https://arxiv.org/abs/2509.24560](https://arxiv.org/abs/2509.24560) |
| 🤗 **Models (Hugging Face)** | [https://huggingface.co/shaohao011/models](https://huggingface.co/shaohao011/models) |
| 🌐 **Project Page** | [https://shaohao011.github.io/AdaThink-Med/](https://shaohao011.github.io/AdaThink-Med/) |

---

## ⚙️ Framework Highlights

| Component | Description |
|------------|--------------|
| **Uncertainty Estimation** | Token-level entropy used to quantify reasoning confidence. |
| **Difficulty Estimation** | Combines correctness and uncertainty to measure problem complexity. |
| **Length Calibration** | Penalizes unnecessary long reasoning on easy problems while encouraging exploration on hard ones. |
| **Adaptive Training** | Two-stage reinforcement learning pipeline for stable and efficient optimization. |

---

## 🧑‍💻 Acknowledgements

We thank the following excellent open-source projects for inspiration and codebases:

- 🪶 [EasyR1](https://github.com/hiyouga/EasyR1)  
- ⚡ [VERL (Volcengine RL Framework)](https://github.com/volcengine/verl)  
- 🩺 [HuatuoGPT-o1](https://github.com/FreedomIntelligence/HuatuoGPT-o1)  
- 🔥 [Entropy Mechanism of RL](https://github.com/PRIME-RL/Entropy-Mechanism-of-RL)

---

## 📝 Citation

If you find **AdaThink-Med** useful for your research, please consider citing our paper:

```bibtex
@article{rui2025adathink,
  title={AdaThink-Med: Medical Adaptive Thinking with Uncertainty-Guided Length Calibration},
  author={Rui, Shaohao and Chen, Kaitao and Ma, Weijie and Wang, Xiaosong},
  journal={arXiv preprint arXiv:2509.24560},
  year={2025}
}
