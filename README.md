# Transformer-VAE

![Diagram of the a python State Autoencoder](https://github.com/Fraser-Greenlee/transformer-vae/blob/master/t-vae.png)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1S8sUSkc_7ON00HDnse1MUXTTflo59VxA?usp=sharing)

Transformer-VAE's learn smooth latent spaces of discrete sequences without any explicit rules in their decoders.

This can be used for program synthesis, drug discovery, music generation and much more!
To lean more about how it works checkout [this blog post](https://fraser-greenlee.github.io/2020/08/13/Transformers-as-Variational-Autoencoders.html).

If you notice any issues please reach out and open an issue! I'll try to get back to you ASAP.

## Install

```bash
git clone https://github.com/Fraser-Greenlee/transformer-vae.git;
cd transformer-vae; python setup.py -q install
```

## Running

The model uses Weights and Biasis for logging. Ensure you have the following enviroment variables set before running:

```bash
WANDB_API_KEY=791c072dd3f0d33aed57e13af6ba86d312cc68c0
WANDB_WATCH=false
WANDB_PROJECT=t-vae_training_demo
```

When running the model will run interpolations and log samples.
Note that this currently doesn't work when using "window attention".

Max run specs (12GB GPU):

* Base model
  * 237seq, 1 batch size
  * 99seq, 7 batch size
  * 30seq, 125 batch size
* Base model (half Funnel 3_3_3)
  * 99seq, 30 batch size
* Base model (tiny Funnel 1_1_1)
  * 237seq 15 batch size
  * 99seq, 40 batch size
* Base model (tiny Funnel 1_1_1) (grad checkpoint every 3 layers)
  * 237seq 30 batch size
* Base model (tiny Funnel 1_1_1) (grad checkpoint every 3 layers + window60)
  * 237seq 50 batch size
* Base model (tiny Funnel 2_2_2) (grad checkpoint every 3 layers + window100-300)
  * 840seq 10 batch size
* Base model (tiny Funnel 1_1_1) (grad checkpoint every layer)
  * 237seq 40 batch size
* Large model,
  * 30seq, 5 batch size

MNIST base:

```bash
!cd transformer-vae; python run_experiment.py mnist_base tenth_5_tkn grad_check_pnt batch_small eval window200 funnel_small
```

Python Lines

```bash
!cd transformer-vae; python run_experiment.py syntax tenth_5_tkn batch_large 30Seq eval
```

News Headlines

```bash
!cd transformer-vae; python run_experiment.py semantics tenth_5_tkn batch_large 30Seq eval
```

Wiki Sentences

```bash
!cd transformer-vae; python run_experiment.py wiki_tokens batch_medium eval 32_latent --dataset_config_name=1M_segment_0
```

## DeepSpeed

For using deepspeed, found model size increase wasn't worth it.

```bash
# install deepspeed
!pip install ninja
!ninja --version
!pip install deepspeed
!ds_report

!pip uninstall -y pyarrow
!pip install --upgrade pyarrow

!pip uninstall -y datasets
!pip install -U datasets
!pip install mpi4py
```