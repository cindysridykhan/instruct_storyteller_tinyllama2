# Training and Fine-Tuning LLM in Python and PyTorch from scratch

This repository contains a standalone low-level Python and PyTorch code to train and LoRA fine-tune a smaller version of Llama2 LLM.
The LLM I trained follows instructions to write tiny stories.

**Demo**
[GIF]

This repository is heavily inspired from Karpathy's [llama2.c repository](https://github.com/karpathy/llama2.c), and for the LoRA part, from wlamond's [PR](https://github.com/karpathy/llama2.c/pull/187).


# Dataset

The dataset I used is the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset, with additional preprocessing steps to rework the prompts. (cf [data_loader.py](data_loader.py)).
