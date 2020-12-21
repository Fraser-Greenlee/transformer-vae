import torch
from torch import nn
from transformers.models.t5.modeling_t5 import T5LayerFF, T5LayerSelfAttention


class LatentEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert(config.transformer.d_model > 100)
        assert(100 * config.set_seq_size > config.latent_size)
        self.shrink_tokens = nn.Linear(config.transformer.d_model, 100)
        self.shrink_sequence = nn.Linear(100 * config.set_seq_size, config.latent_size)
        self.tanh = nn.Tanh()

    def forward(self, encoding) -> torch.Tensor:
        batch_size = encoding.size(0)
        encoding = self.shrink_tokens(encoding)
        encoding = self.shrink_sequence(encoding.view(batch_size, -1))
        return self.tanh(encoding)


class LatentEncoder1stToken(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert(config.transformer.d_model > config.latent_size)
        self.token_to_latent = nn.Linear(config.transformer.d_model, config.latent_size)
        self.tanh = nn.Tanh()

    def forward(self, encoding) -> torch.Tensor:
        return self.tanh(self.token_to_latent(encoding[:, 0, :]))


class LatentEncoderFull1stToken(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.transformer.d_model == config.latent_size

    def forward(self, encoding) -> torch.Tensor:
        return encoding[:, 0, :]


class LatentEncoderFullNTokens(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.transformer.d_model * config.n_latent_tokens == config.latent_size
        self.n_tokens = config.n_latent_tokens

    def forward(self, encoding) -> torch.Tensor:
        batch_size = encoding.size(0)
        return encoding[:, :self.n_tokens, :].view(batch_size, -1)


class LatentEncoderAttention(LatentEncoder):
    """
    Uses attention on token-encodings before compressing them.

    Should weight each individual token's expected importance for the overall sequence representation.
    """
    def __init__(self, config):
        super().__init__(config)
        self.token_scorer = nn.Linear(config.transformer.d_model, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoding) -> torch.Tensor:
        batch_size = encoding.size(0)
        token_scores = self.softmax(self.token_scorer(encoding))
        encoding = self.shrink_tokens(encoding * token_scores)
        encoding = self.shrink_sequence(encoding.view(batch_size, -1))
        return self.tanh(encoding)


class LatentDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.decode_latent = nn.Linear(config.latent_size, 10 * config.set_seq_size)
        self.grow_sequence = nn.Linear(10 * config.set_seq_size, 100 * config.set_seq_size)
        self.grow_tokens = nn.Linear(100, config.transformer.d_model)

        if config.model_type == "t5":
            # TODO would this actually effect `dropout_rate`?
            dropout_rate = config.dropout_rate
            config.dropout_rate = 0
            self.norm = T5LayerFF(config)
            config.dropout_rate = dropout_rate
        elif config.model_type == "funnel":
            self.norm = nn.LayerNorm(config.transformer.d_model, config.layer_norm_eps)
        else:
            raise ValueError(f'Unknown config.model_type "{config.model_type}"')

    def forward(self, latent) -> torch.Tensor:
        batch_size = latent.size(0)
        latent = self.decode_latent(latent)
        latent = self.grow_sequence(latent)
        return self.norm(
            self.grow_tokens(
                latent.view(batch_size, -1, 100)
            )
        )


class LatentDecoderSingleToken(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.decode_latent = nn.Linear(config.latent_size, 100)
        self.grow_token = nn.Linear(100, config.transformer.d_model)
        self.dim_m = config.transformer.d_model

        if config.model_type == "t5":
            config.dropout_rate = 0
            self.norm = T5LayerFF(config)
        elif config.model_type == "funnel":
            self.norm = nn.LayerNorm(config.transformer.d_model, config.layer_norm_eps)
        else:
            raise ValueError(f'Unknown config.model_type "{config.model_type}"')

    def forward(self, latent) -> torch.Tensor:
        batch_size = latent.size(0)
        latent = self.decode_latent(latent)
        return self.norm(self.grow_token(latent).view(batch_size, -1, self.dim_m))


class LatentDecoderFullSingleToken(nn.Module):
    def __init__(self, config):
        assert config.transformer.d_model <= config.latent_size
        self.dim_m = config.transformer.d_model
        super().__init__()

    def forward(self, latent) -> torch.Tensor:
        batch_size = latent.size(0)
        return latent.view(batch_size, -1, self.dim_m)


class LatentDecoderMatchEncoder(LatentDecoder):
    """
    Just do one jump from latent -> 100x sequence.
    """

    def __init__(self, config):
        super().__init__(config)
        self.grow_sequence = nn.Linear(config.latent_size, 100 * config.set_seq_size)

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
    "1st-token": LatentEncoder1stToken,
    "full-1st-token": LatentEncoderFull1stToken,
    "full-n-tokens": LatentEncoderFullNTokens,
    "basic-attention": LatentEncoderAttention,
}
VAE_DECODER_MODELS = {
    None: LatentDecoder,
    "single-token": LatentDecoderSingleToken,
    "full-single-token": LatentDecoderFullSingleToken,
    "match-encoder": LatentDecoderMatchEncoder,
    "attention": LatentDecoderSelfAttnGrow,
}
