import logging
import math
import torch
from torch import nn
from transformers.models.t5.modeling_t5 import T5LayerFF, T5LayerSelfAttention


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


class LatentEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.transformer.d_model > 100
        assert 100 * config.transformer.n_positions > config.latent_size
        self.shrink_tokens = nn.Linear(config.transformer.d_model, 100)
        self.shrink_sequence = nn.Linear(100 * config.transformer.n_positions, config.latent_size)
        self.tanh = nn.Tanh()

    def forward(self, encoding) -> torch.Tensor:
        batch_size = encoding.size(0)
        encoding = self.shrink_tokens(encoding)
        encoding = self.shrink_sequence(encoding.view(batch_size, -1))
        return self.tanh(encoding)


class LatentEncoderNTokens(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.new_token_dim = math.ceil(config.latent_size / config.n_latent_tokens)
        if self.new_token_dim == config.transformer.d_model:
            logger.info('Latent encoder is not rescaling the latent code.')
        self.token_to_latent = nn.Linear(config.transformer.d_model, self.new_token_dim)
        self.n_tokens = config.n_latent_tokens
        self.tanh = nn.Tanh()

    def forward(self, encoding) -> torch.Tensor:
        batch_size = encoding.size(0)
        return self.tanh(self.token_to_latent(encoding))[:, : self.n_tokens, :].view(batch_size, -1)


class LatentDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        t_config = config.transformer

        self.decode_latent = nn.Linear(config.latent_size, 10 * t_config.n_positions)
        self.grow_sequence = nn.Linear(10 * t_config.n_positions, 100 * t_config.n_positions)
        self.grow_tokens = nn.Linear(100, t_config.d_model)

        if t_config.model_type == "t5":
            # TODO would this actually effect `dropout_rate`?
            dropout_rate = t_config.dropout_rate
            t_config.dropout_rate = 0
            self.norm = T5LayerFF(t_config)
            t_config.dropout_rate = dropout_rate
        elif t_config.model_type == "funnel":
            self.norm = nn.LayerNorm(t_config.d_model, t_config.layer_norm_eps)
        else:
            raise ValueError(f'Unknown config.transformer.model_type: "{t_config.model_type}"')

    def forward(self, latent) -> torch.Tensor:
        batch_size = latent.size(0)
        latent = self.decode_latent(latent)
        latent = self.grow_sequence(latent)
        return self.norm(self.grow_tokens(latent.view(batch_size, -1, 100)))


class LatentDecoderNTokens(nn.Module):
    '''
        Convert multiple token encodings into a single latent.
    '''
    def __init__(self, config):
        super().__init__()
        self.latent_token_dim = math.ceil(config.latent_size / config.n_latent_tokens)
        if self.latent_token_dim == config.transformer.d_model:
            logger.info('Latent decoder is not rescaling the latent code.')
            self.latent_to_token = lambda x: x
        else:
            self.latent_to_token = nn.Linear(self.latent_token_dim, config.transformer.d_model)

    def forward(self, latent) -> torch.Tensor:
        batch_size = latent.size(0)
        return self.latent_to_token(latent.view(batch_size, -1, self.latent_token_dim))


class LatentDecoderMatchEncoder(LatentDecoder):
    """
    Just do one jump from latent -> 100x sequence.
    """

    def __init__(self, config):
        super().__init__(config)
        self.grow_sequence = nn.Linear(config.latent_size, 100 * config.transformer.n_positions)

    def forward(self, latent) -> torch.Tensor:
        batch_size = latent.size(0)
        latent = self.grow_sequence(latent)
        return self.norm(self.grow_tokens(latent.view(batch_size, -1, 100)))


class LatentDecoderSelfAttnGrow(LatentDecoder):
    """
    Start with 10-dim tokens and grow them whith cross-attention.
    """

    # TODO add position bias
    def __init__(self, config):
        super().__init__(config)
        self.grow_tokens_to_100 = nn.Linear(10, 100)
        self.grow_tokens_to_dim_m = nn.Linear(100, config.dim_m)
        config.transformer.d_model = 100
        self.self_attention = T5LayerSelfAttention(config.transformer)

    def forward(self, latent) -> torch.Tensor:
        batch_size = latent.size(0)
        latent = self.grow_sequence(latent)
        encoding_100_dim = self.self_attention(self.norm(self.grow_tokens_to_100(latent.view(batch_size, -1, 10))))
        return self.norm(self.grow_tokens_to_dim_m(encoding_100_dim))


VAE_ENCODER_MODELS = {
    None: LatentEncoder,
    "n-tokens": LatentEncoderNTokens,
}
VAE_DECODER_MODELS = {
    None: LatentDecoder,
    "n-tokens": LatentDecoderNTokens,
    "match-encoder": LatentDecoderMatchEncoder,
    "attention": LatentDecoderSelfAttnGrow,
}
