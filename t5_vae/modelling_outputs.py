from dataclasses import dataclass
from typing import Optional
import torch
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import Seq2SeqLMOutput


@dataclass
class BaseVAEOutput(ModelOutput):
    """
    Base class for VAE's outputs, with latent codes & encoding.

    Args:
        reconstructed_encoding (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Reconstructed hidden states originally from the last layer of the encoder.
        latent_code (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, latent_size)`):
            Latent codes representing encoded sequences.
        reg_loss (:obj:`torch.FloatTensor` of shape :obj:`(batch_size)`):
            MMD-VAE regularisation loss for this step.
    """

    latent_code: torch.FloatTensor
    reconstructed_encoding: torch.FloatTensor
    reg_loss: torch.FloatTensor


@dataclass
class VAE_Seq2SeqLMOutput(Seq2SeqLMOutput):
    """
    Seq2SeqLMOutput extended to include VAE-specific attributed latent_code & reg_loss.

    Args:
        reconstructed_encoding (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Reconstructed hidden states originally from the last layer of the encoder.
        latent_code (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, latent_size)`):
            Latent codes representing encoded sequences.
        reg_loss (:obj:`torch.FloatTensor` of shape :obj:`(batch_size)`):
            MMD-VAE regularisation loss for this step.
        reg_loss (:obj:`torch.FloatTensor` of shape :obj:`(batch_size)`):
            MMD-VAE regularisation loss for this step.
    """

    latent_code: torch.FloatTensor = None
    reconstructed_encoding: torch.FloatTensor = None
    reg_loss: torch.FloatTensor = None
    decoder_ce: torch.FloatTensor = None
