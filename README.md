# Information Extraction with Fine-Tuned Language Models

An NLP project focused on fine-tuning open-source language models to extract structured information from unstructured text and guarantee valid JSON outputs.

---

## Project Overview

Information extraction is a critical NLP task that transforms raw text into structured formats for downstream systems.

This project fine-tunes a pretrained model to extract key entities, including:
- Organizations
- Locations
- Events
- Dates

### Example

**Input:**
```text
OpenAI announced a new event in San Francisco on October 20, 2026.

Target Output:
JSON

{
  "organization": "OpenAI",
  "location": "San Francisco",
  "event": "a new event",
  "date": "October 20, 2026"
}

    Note: This is an illustrative example of the target schema, not a measured inference result.

Key Objectives

    Curate and preprocess a domain-specific dataset for supervised fine-tuning (SFT).

    Apply Parameter-Efficient Fine-Tuning (PEFT / LoRA) to fit within local compute constraints.

    Evaluate extraction accuracy alongside JSON syntax validity rate.

    Build an end-to-end reproducible pipeline for training and inference.

Technology Stack

    Core: Python, PyTorch

    Ecosystem: Hugging Face (transformers, datasets, peft, accelerate, bitsandbytes)

    Version Control: Git & GitHub

Development Environment

    GPU: NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM)

    RAM: 48 GB

    Strategy: 4-bit/8-bit quantization (QLoRA) and gradient accumulation to ensure stable execution within 6 GB VRAM.

Project Status & Roadmap

Status: In Progress

    [x] Environment setup (Python, PyTorch with CUDA acceleration)

    [x] Repository initialization & dependency tracking

    [ ] Base model selection

    [ ] SFT dataset preparation and formatting

    [ ] LoRA / QLoRA training pipeline implementation

    [ ] Evaluation & schema validation benchmarking

    [ ] Inference pipeline & documentation

Repository Structure
Plaintext

information-extraction-ft/
├── data/              # Preprocessed datasets (ignored by git)
├── notebooks/         # Exploratory data analysis & prototyping
├── outputs/           # Training logs and adapters (ignored by git)
├── src/               # Modular training, preprocessing, and inference code
├── .gitignore
├── requirements.txt
└── README.md

Evaluation Metrics

Model performance will be evaluated on unseen test data using:

    Schema Integrity: JSON validity rate (parsable without errors).

    Extraction Accuracy: Exact Match (EM), Precision, Recall, and F1-score per field.

    Inference Latency: VRAM footprint and token generation speed.

Future Improvements

    Experiment with structured generation libraries (such as Guidance, Outlines, or Instructor).

    Serve the fine-tuned adapter via a FastAPI endpoint.

    Compare QLoRA results against few-shot prompting on larger baseline models.

Disclaimer

This project is developed for educational and experimental purposes, focusing on local parameter-efficient fine-tuning workflows and structured output extraction.