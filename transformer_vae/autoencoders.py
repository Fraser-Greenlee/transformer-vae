import logging
import torch
from torch import nn
from transformers.models.t5.modeling_t5 import T5LayerFF

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


class LatentEncoderNTokens(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_to_latent = nn.Linear(config.transformer.d_model, config.latent_token_dim)
        self.n_tokens = config.n_latent_tokens
        self.tanh = nn.Tanh()

    def forward(self, encoding) -> torch.Tensor:
        batch_size = encoding.size(0)
        return self.tanh(self.token_to_latent(encoding))[:, : self.n_tokens, :].view(batch_size, -1)


class LatentDecoderNTokens(nn.Module):
    '''
        Convert multiple token encodings into a single latent.
    '''
    def __init__(self, config):
        super().__init__()
        self.latent_token_dim = config.latent_token_dim
        if self.latent_token_dim == config.transformer.d_model:
            logger.info('Latent decoder is not rescaling the latent code.')
            self.latent_to_token = lambda x: x
        else:
            self.latent_to_token = nn.Linear(self.latent_token_dim, config.transformer.d_model)

    def forward(self, latent) -> torch.Tensor:
        batch_size = latent.size(0)
        return self.latent_to_token(latent.view(batch_size, -1, self.latent_token_dim))


class LatentDecoderT5Norm(LatentDecoderNTokens):
    '''
        Use T5 norm.
    '''
    def __init__(self, config):
        super().__init__(config)
        t_config = config.transformer_decoder
        dropout_rate = t_config.dropout_rate
        t_config.dropout_rate = 0
        self.norm = T5LayerFF(t_config)
        t_config.dropout_rate = dropout_rate

    def forward(self, latent) -> torch.Tensor:
        return self.norm(super().forward(latent))


class LatentDecoderFunnelNorm(LatentDecoderNTokens):
    '''
        Use Funnel norm.
    '''
    def __init__(self, config):
        super().__init__(config)
        t_config = config.transformer
        self.norm = nn.LayerNorm(t_config.d_model, t_config.layer_norm_eps)

    def forward(self, latent) -> torch.Tensor:
        return self.norm(super().forward(latent))


VAE_ENCODER_MODELS = {
    None: LatentEncoderNTokens,
}
VAE_DECODER_MODELS = {
    None: LatentDecoderNTokens,
    "t5_norm": LatentDecoderT5Norm,
    "funnel_norm": LatentDecoderFunnelNorm,
}
