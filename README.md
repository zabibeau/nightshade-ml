# Nightshade

This repo contains code and information for a project for CS 5806

## Project Details
In 2023 Shan et al. created Nightshade, which is an adversarial poisoning framework that takes an artist's image and adds imperceptible perturbations. The aim is to hinder an image generative model's ability to learn from artist's work or to reproduce them through these perturbations. Nightshade is a strong step towards protecting artists' work, but their implementation relies on a single iterative penalty method, which may be a limitation. In order to strongly poison an image, it often may require large perturbations which risks causing visible artifacts to appear on the image.

We attempted to mix adversarial attacks like Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) into the Nightshade pipeline, hoping to create a strong and still imperceptible image poisoning result. We applied these methods within the VAE latent space of stable diffusion instead of working directly on pixel values. 

## Links to Paper and Presentation Slides
https://drive.google.com/file/d/1zUQK7B1vOrR4kt_dyi1WjiHCzvSX9vJx/view?usp=sharing
https://drive.google.com/file/d/1bokBLzdNwx9VvkWeQ5n6YZe9lOZ85r80/view?usp=sharing

## Poisoning Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zabibeau/nightshade-ml/blob/main/nightshade_poisoning.ipynb)

## Training Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zabibeau/nightshade-ml/blob/main/train_sd.ipynb)
## Evaluation Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zabibeau/nightshade-ml/blob/main/evaluate_nightshade.ipynb)

## Setup Instructions
1. Install python if you haven't (use any version except for 3.13)
2. Run `python3 -m venv nightshade_env` in a terminal
3. Run `source nightshade_env/bin/activate` to activate the venv, and `deactivate` to deactivate it
4. Install pytorch by going to: [https://pytorch.org/get-started/locally/]
5. To install all requirements, run `pip3 install -r requirements.txt`
6. To download all training data, run `python3 download_data.py`. This will take a little bit, as we're downloading data from the 2014 Coco dataset (it's around 13GB)

