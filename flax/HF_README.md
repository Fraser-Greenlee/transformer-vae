# Transformer-VAE

Convert a T5 model into a variational autoencoder for text.

I have already made a project that does this in PyTorch.

This project is to convert it into Flax so it can be trained efficiently on a TPU to train the largest every Transformer-VAE!

## Language

The model will be trained in english.

## Model

T5-base, this will match with the Optimus model.

## Datasets

Use the wikipedia sentences dataset from OPTIMUS.

This comes tokenized so we'll need to use its tokenizer with T5.

https://github.com/ChunyuanLI/Optimus/blob/master/download_datasets.md

## Training scripts

The original [PyTorch training script](https://github.com/Fraser-Greenlee/transformer-vae/blob/master/transformer_vae/train.py) is adapted from the old Huggingface clm training script so using the [flax clm script](https://github.com/huggingface/transformers/blob/master/examples/flax/language-modeling/run_clm_flax.py) should be a good base to build on.

## Challenges

It can be hard to know the right model hyperparameters for a Transformer-VAE, particularly as it scales. The OPTIMUS results and my old runs will be a helpful starting point.

The original model has been made in PyTorch so there will be some features that can't be ported over.

## Desired project outcome

A colab notebook where people can explore the Transformer-VAE's latent space.

## Reads

Here are some background links to understand the context behind this project:

- [Initial Transformer-VAE post](http://fras.uk/ml/large%20prior-free%20models/transformer-vae/2020/08/13/Transformers-as-Variational-Autoencoders.html)
 and the [improvements post](http://fras.uk/ml/large%20prior-free%20models/transformer-vae/2021/02/23/An-Improved-Transformer-VAE.html).
- [MMD-VAE](https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/), the MMD loss that Transformer-VAE uses.
- [OPTIMUS](https://www.microsoft.com/en-us/research/publication/optimus-organizing-sentences-via-pre-trained-modeling-of-a-latent-space/) Current SOTA text-vae will give a sense of the outputs would should expet.
