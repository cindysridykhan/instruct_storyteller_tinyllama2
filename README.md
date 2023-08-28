# Training and Fine-Tuning LLM in Python and PyTorch from scratch

This repository contains a standalone low-level Python and PyTorch code to train and LoRA fine-tune a smaller version of Llama2 LLM.
The LLM I trained follows instructions to write tiny stories.

**Demo**

<img src="/story1500.gif" width="500" height="500"/>

This repository is heavily inspired from Karpathy's [llama2.c repository](https://github.com/karpathy/llama2.c), and for the LoRA part, from wlamond's [PR](https://github.com/karpathy/llama2.c/pull/187).


# Dataset

The dataset I used is the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset, with additional preprocessing steps to rework the prompts. (cf [data_loader.py](data_loader.py)).

# Training from scratch

Training from scratch can be done from the notebook [instruct_training_from_scratch.ipynb](instruct_training_from_scratch.ipynb).

# LoRA Fine-tuning

LoRA Fine-tuning can be done from the notebook [instruct_finetuning.ipynb](instruct_finetuning.ipynb). 
By default in this notebook, I started from Karpathy's 110M parameters pretrained model that you can find on HuggingFace Hub at [tinyllamas](https://huggingface.co/karpathy/tinyllamas). 
LoRA is then applied to the architecture, with rank 2 matrices and on ['wq', 'wk', 'wo', 'wv'] layers.


# Evaluation

You can try the models using the generate.ipynb notebook.
