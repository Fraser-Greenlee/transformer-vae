
# Flax

Here is where the new Flax code will live!

Training logs will write to https://wandb.ai/fraser/flax-vae

## ToDo

- [ ] Convert `transformers/examples/flax/language-modeling/run_clm_flax.py` into a new training script for transformer-VAE's.
  - Use an "empty VAE"  a.k.a just sends the encoding to the decoder with no regularisation loss, use the T5 encoder & decoder.
- [ ] Make a `autoencoders.py` version of `autoencoders.py`.
  - Start with turning the last model into a valid VAE.
- [ ] Add support for training performance logs.
- [ ] Train on the OPTIMUS wikipedia sentences dataset.
  - [ ] Make a tokenizer using the OPTIMUS tokenized dataset.
- [ ] Try interpolating sentences!
