# Training and Fine-Tuning LLM in Python and PyTorch from scratch

This repository contains a standalone low-level Python and PyTorch code to train and fine-tune a small version of the Llama2 LLM.
The LLM I trained follows instructions to write tiny stories.

**Demo**

<img src="/story1500.gif" width="500" height="500"/>

This repository is heavily inspired from Karpathy's [llama2.c repository](https://github.com/karpathy/llama2.c), and for the LoRA part, from wlamond's [PR](https://github.com/karpathy/llama2.c/pull/187).

## Installation
#### Requirements
Install requirements in your environment:
```
pip install -r requirements.txt
```
#### Models

The models are available on HuggingFace hub:
- [Trained from scratch model (15M parameters)]
- [LoRA finetuned model (110M parameters)]


## Inference

```
python generate.py --model_path='./models/lora_story_teller_110M.pt' --prompt='Write a story. In the story, try to use the verb "climb", the noun "ring" and the adjective "messy". Possible story:' --temperature=0.1 --top_k=10
```
By default, parameters are temperature = 0.5 and top_k = 10.

Alternatively, you can also use the [generate.ipynb](notebooks/generate.ipynb) notebook.

## Training

Note: All the following .py files are available in [notebook version](notebooks)

### Dataset

The dataset I used is the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset, with additional preprocessing steps to rework the prompts. (see [blogpost]() for more details).
To prepare the dataset, run
```
python prepare_dataset.py --input_dir='path/to/TinyStoriesFolder' --output_dir='data'
```

### Training from scratch

Training from scratch can be done from the notebook [instruct_training_from_scratch.ipynb](instruct_training_from_scratch.ipynb).

### LoRA Fine-tuning

LoRA Fine-tuning can be done from the notebook [instruct_finetuning.ipynb](instruct_finetuning.ipynb). 
By default in this notebook, I started from Karpathy's 110M parameters pretrained model that you can find on HuggingFace Hub at [tinyllamas](https://huggingface.co/karpathy/tinyllamas). 
LoRA is then applied to the architecture, with rank 2 matrices and on ['wq', 'wk', 'wo', 'wv'] layers.



## Notes on the trained models
Currently, the models only support prompts like 'Write a story. In the story, try to use the verb "{verb}", the noun "{noun}" and the adjective "{adj}". The story has the following features: it should contain a dialogue. Possible story:', that is, prompts that look like the one in the training set. Plus, in order for the story to make sens, the verb, noun and adjective given must be common words that are present in the training set.

This is because it has been trained only on the TinyStories dataset. It would be interesting to make the dataset more diverse, or to finetune from one of the pretrained llama2 models.
