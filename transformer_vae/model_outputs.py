from dataclasses import dataclass
import torch
from typing import Tuple, List, Optional

from transformers.file_utils import ModelOutput


@dataclass
class BaseVAE_Output(ModelOutput):
    """
    Base class for a VAE's outputs.

    Args:
        reconstructed_encoding (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Reconstructed hidden states originally from the last layer of the encoder.
        latent (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, latent_size)`):
            Latent codes representing encoded sequences.
        reg_loss (:obj:`torch.FloatTensor` of shape :obj:`(batch_size)`):
            MMD-VAE regularisation loss for this step.
    """

    latent: torch.FloatTensor = None
    reconstructed_encoding: torch.FloatTensor = None
    reg_loss: Optional[torch.FloatTensor] = None


@dataclass
class BaseTransformerVAE_Output(ModelOutput):
    """
    Base class for a Transformer-VAE's outputs.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Language modeling loss.
        reconstructed_encoding (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Reconstructed hidden states originally from the last layer of the encoder.
        latent (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, latent_size)`):
            Latent codes representing encoded sequences.
        reg_loss (:obj:`torch.FloatTensor` of shape :obj:`(batch_size)`):
            MMD-VAE regularisation loss for this step.
        reg_loss (:obj:`torch.FloatTensor` of shape :obj:`(batch_size)`):
            MMD-VAE regularisation loss for this step.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

    latent: torch.FloatTensor = None
    reconstructed_encoding: Optional[torch.FloatTensor] = None
    reg_loss: Optional[torch.FloatTensor] = None
    decoder_ce: Optional[torch.FloatTensor] = None
