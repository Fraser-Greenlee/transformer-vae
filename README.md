# T5-VAE

![Diagram of the a State Autoencoder](t5-vae.png)

T5-VAE learns smooth latent spaces of sequences without hard-coding rules into the decoder.

This could be used for program synthesis, drug discovery and much more!

To see how it works checkout [this blog post]().

## Training

First get a text file with each line representing a training sample.
The default script uses [Weights & Biasis](https://app.wandb.ai/) for logging, see old runs [here](https://app.wandb.ai/fraser/T5-VAE?workspace=user-fraser).

Edit the `train_t5_vae.sh` script & run with `bash train_t5_vae.sh`.

To explore the produced latent space open it using `Colab_T5_VAE.ipynb`.
