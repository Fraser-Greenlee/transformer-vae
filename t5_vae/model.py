"""
    Define the T5-VAE model & all its variations.
"""
from dataclasses import dataclass, field
from typing import Optional
import torch
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_t5 import T5LayerFF


@dataclass
class T5_VAE_ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    t5_model_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the T5 model being using for encoding & decoding."},
    )
    # TODO allow using Funnel-Transformer here
    """
    base_model_type: Optional[str] = field(
        default="t5",
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    """
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    ae_latent_size: int = field(
        default=None, metadata={"help": "The size of the VAE's latent space, only valid with a T5 model."}
    )
    set_seq_size: int = field(default=None, metadata={"help": "Set sequence size."})
    # args used during training
    reg_schedule_k: float = field(
        default=0.0025,
        metadata={"help": "Multiplied by global_step in a sigmoid, more gradually increase regulariser loss weight."},
    )
    reg_schedule_b: float = field(
        default=6.25,
        metadata={"help": "Added to global step in sigmoid, further delays increase in regulariser loss weight."},
    )
    reg_constant_weight: Optional[float] = field(
        default=None, metadata={"help": "Apply a constant weight to the regulariser."}
    )


class LatentEncoderLargeTanh_1kLatent(nn.Module):
    def __init__(self, dim_m, set_input_size, latent_size):
        super().__init__()
        assert dim_m > 100
        self.shrink_tokens = nn.Linear(dim_m, 100)
        self.shrink_sequence = nn.Linear(100 * set_input_size, latent_size)
        self.tanh = nn.Tanh()

    def forward(self, encoding) -> torch.Tensor:
        batch_size = encoding.size(0)
        # shrink each tokens encoding
        encoding = self.shrink_tokens(encoding)
        encoding = self.shrink_sequence(encoding.view(batch_size, -1))
        return self.tanh(encoding)


class LatentDecoderLargeT5NormFF(nn.Module):
    def __init__(self, dim_m, set_input_size, latent_size, config):
        super().__init__()
        self.decode_latent = nn.Linear(latent_size, 10 * set_input_size)
        self.grow_sequence = nn.Linear(10 * set_input_size, 100 * set_input_size)
        self.grow_tokens = nn.Linear(100, dim_m)

        old_drop = config.dropout_rate
        config.dropout_rate = 0
        self.norm = T5LayerFF(config)
        config.dropout_rate = old_drop

    def forward(self, latent) -> torch.Tensor:
        batch_size = latent.size(0)
        # grow each tokens encoding
        latent = self.decode_latent(latent)
        latent = self.grow_sequence(latent)
        return self.norm(self.grow_tokens(latent.view(batch_size, -1, 100)))


class EncoderDecoderVAE(nn.Module):
    """
    An MMD-VAE used with encoder-decoder models.
    Encodes all token encodings into a single latent & spits them back out.
    """

    def __init__(self, encoder, decoder, training_args):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.args = training_args

    def _model_forward(self, encoding):
        latent = self.encoder(encoding)
        return self.decoder(latent), latent

    def forward(self, input_encoding: torch.Tensor, just_get_latent=False, just_get_encoding=False):
        recon_encoding, latent = self._model_forward(input_encoding)
        if just_get_latent:
            return latent
        if just_get_encoding:
            return recon_encoding
        recon_loss = torch.nn.MSELoss(reduction="mean")(input_encoding, recon_encoding)
        reg_loss = self._regularliser_loss(input_encoding, latent)
        return recon_loss, reg_loss, recon_encoding

    @staticmethod
    def _compute_kernel(x, y):
        x_size = x.shape[0]
        y_size = y.shape[0]
        dim = x.shape[1]

        tiled_x = x.view(x_size, 1, dim).repeat(1, y_size, 1)
        tiled_y = y.view(1, y_size, dim).repeat(x_size, 1, 1)

        return torch.exp(-torch.mean((tiled_x - tiled_y) ** 2, dim=2) / dim * 1.0)

    def _compute_mmd(self, x, y):
        x_kernel = self._compute_kernel(x, x)
        y_kernel = self._compute_kernel(y, y)
        xy_kernel = self._compute_kernel(x, y)
        return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)

    def _regularliser_loss(self, input_encoding, latent):
        loss = torch.tensor(0, dtype=torch.float).to(self.args.device)
        true_samples = torch.randn(latent.size()).to(latent.device)
        loss += self._compute_mmd(true_samples, latent)
        return loss


class t5_VAE(PreTrainedModel):
    base_model_prefix = "t5_vae"

    def __init__(self, config, t5_model, vae, set_seq_size, tokenizer):
        super().__init__(config=config)
        self.t5_model = t5_model
        self.vae = vae
        self.config = config
        self.set_seq_size = set_seq_size
        self.tokenizer = tokenizer

    def pad_input_ids(self, input_ids):
        padedd_input_tokens = torch.ones(self.set_seq_size, dtype=torch.long) * self.tokenizer.pad_token_id
        padedd_input_tokens[: input_ids.size(0)] = input_ids
        return padedd_input_tokens.to("cuda").view(1, -1)

    def _decoder_logits(self, decoder_input_ids, encoding):
        sequence_output = self.t5_model.decoder(input_ids=decoder_input_ids, encoder_hidden_states=encoding)[0]
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.t5_model.model_dim ** -0.5)
        logits = self.t5_model.lm_head(sequence_output)
        return logits

    def decoder_loss(self, labels, encoding, ignore_index=-100):
        decoder_input_ids = self.t5_model._shift_right(labels)
        logits = self._decoder_logits(decoder_input_ids, encoding)
        loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none")
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
        return loss

    def decoder_loss_from_latent(self, labels, latent):
        encoding = self.vae.decoder(latent)
        return self.decoder_loss(labels, encoding)

    def get_latent(self, input_ids):
        attention_mask = input_ids.ne(self.config.pad_token_id).long()
        encoding = self.t5_model.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        return self.vae(encoding, just_get_latent=True)

    def get_hidden(self, input_ids):
        attention_mask = input_ids.ne(self.config.pad_token_id).long()
        encoding = self.t5_model.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        return self.vae(encoding, just_get_encoding=True)

    def _greedy_logits(self, encoding):
        # always start with 0 token
        decoder_input_ids = torch.tensor([[0]]).to(self.device)
        for i in range(self.set_seq_size):
            logits = self._decoder_logits(decoder_input_ids, encoding)
            _, chosen_token = torch.topk(logits[0, i], 1)  # get index of max logits[-1]
            if chosen_token == self.tokenizer.eos_token_id:
                break
            decoder_input_ids = torch.cat((decoder_input_ids, chosen_token.view(1, -1)), 1)
        return logits

    def greedy_logits(self, input_ids=None, latent=None, encoding=None):
        # get logits for given input_ids or latent
        assert input_ids is not None or latent is not None or encoding is not None
        if encoding is None and latent is None:
            if len(input_ids.size()) == 1 and input_ids.size(0) < self.set_seq_size:
                input_ids = self.pad_input_ids(input_ids)
            latent = self.get_latent(input_ids)
        if encoding is None:
            encoding = self.vae.decoder(latent.view(1, -1))
        return self._greedy_logits(encoding)

    def forward(self, input_ids):
        attention_mask = input_ids.ne(self.config.pad_token_id).long()

        encoding = self.t5_model.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        recon_loss, reg_loss, encoding = self.vae(encoding)
        decoder_ce = self.decoder_loss(input_ids, encoding, ignore_index=self.config.pad_token_id)

        return decoder_ce, recon_loss, reg_loss
