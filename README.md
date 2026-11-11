# The Role of Recurrent Refinement in Medical Image Segmentation

This repository presents the original implementation of the paper [The Role Of Recurrent Refinement In Medical Image Segmentation](https://arxiv.org/html/2611.09436v1) by João D.S. Marques, David Calhas and Arlindo Oliveira.

## Overview 📚
While feed-forward segmentation networks perform well on large cohorts, they degrade in constrained clinical settings with few training samples and imaging artifacts. Human vi- sion, by contrast, remains robust under these conditions, a property partly attributed to the recurrent and feedback con- nectivity of the visual cortex, which refines visual information over time. Inspired by this, we append lightweight recurrent refinement blocks to segmentation backbones, namely Condi- tional Random Fields (CRFs), modern Hopfield networks and convolutional GRUs, and study their behaviour as the number of iterations increases at test time (including beyond the train- ing horizon). Evaluated on two open-source datasets, Kvasir- SEG and ARCADE, our recurrent blocks add less than 1.5% parameters and yield gains on clean data (up to 1.4 pp) and under noise (up to 17 pp) while reducing HD95.

## Datasets 📝

We use 2 datasets for this research [KVASIR-Seg](https://www.kaggle.com/datasets/debeshjha1/kvasirseg) and [ARCADE](https://zenodo.org/records/10390295). 

## How to run 💻

To use [WandB](https://wandb.ai) , you can run the main file with the `--wandb` flag:

```bash
python3 run_recurrent_segmentation.py
```
Note: Make sure to add your Weights & Biases (wandb) API key in the main code or configure it using wandb login.

## Citation 💬
```bibtex
@misc{marques2026_recurrentRefinementSegmentation,
  title         = {The Role Of Recurrent Refinement In Medical Image Segmentation}, 
  author        = {Joao D. S. Marques and David Calhas and Arlindo L. Oliveira},
  year          = {2026},
  eprint        = {2611.09436},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {[https://arxiv.org/abs/2611.09436](https://arxiv.org/abs/2611.09436)}
}
