---
title: "AI Research Integrity Officer"
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: docker
sdk_version: "3.11"
python_version: "3.11"
app_file: space_app.py
pinned: false
license: mit
---

# 🔬 AI Research Integrity Officer – Scientific Fraud Detection Environment

Real-world environment where an AI agent investigates fabricated data. Three tasks (easy → hard), dense rewards, programmatic graders.

## Live Demo
[![Hugging Face Spaces](https://img.shields.io/badge/🤗-Spaces-blue)](https://huggingface.co/spaces/your-username/your-space)

## Tasks
| Difficulty | Fabrication Type | Real‑world case |
|------------|------------------|------------------|
| Easy | Duplicate rows | Marc Hauser (Harvard) |
| Medium | Benford violation | Yoshitaka Fujii (200+ papers) |
| Hard | Impossible correlation + timestamp reuse | Ranga Dias (superconductor) |

## Setup
```bash
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama3-70b-8192"
export HF_TOKEN="your_groq_key"
python inference.py