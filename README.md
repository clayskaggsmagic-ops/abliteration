# Abliteration on Qwen 2.5

> **Remove refusal behavior from LLMs via orthogonalization — no training required.**

This project implements the **abliteration** technique on Qwen 2.5 models, allowing you to surgically remove the "refusal direction" that causes instruction-tuned models to decline harmful requests.

## 🎓 Educational Project

This repository is designed for learning about:
- Transformer mechanistic interpretability
- How safety behaviors are encoded in LLMs
- Representation engineering techniques

## 🚀 Quick Start

```bash
# 1. Clone and install dependencies
cd Abliteration
pip install -r requirements.txt

# 2. Run abliteration on Qwen 2.5 0.5B
python run_abliteration.py --model Qwen/Qwen2.5-0.5B-Instruct

# 3. Test the abliterated model
python run_abliteration.py --test --model ./output/abliterated_model
```

## 📖 Documentation

- [**01_theory.md**](docs/01_theory.md) — Understanding how abliteration works
- [**02_implementation.md**](docs/02_implementation.md) — Code walkthrough
- [**03_results.md**](docs/03_results.md) — Results documentation

## 🔧 Requirements

- Python 3.10+
- ~4GB RAM (for 0.5B model)
- Hugging Face account (for model access)

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. Removing safety guardrails from AI models can enable harmful outputs. Use responsibly.

## 📚 References

- [Refusal in LLMs is Mediated by a Single Direction](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ)
- [failspy's Ortho Cookbook](https://huggingface.co/failspy/llama-3-70B-Instruct-abliterated)
- [TransformerLens Documentation](https://neelnanda-io.github.io/TransformerLens/)
