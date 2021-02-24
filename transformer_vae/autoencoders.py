import copy
import torch
from torch import nn
from torch_dct import dct, idct
from transformers.utils import logging

from transformer_vae.model_outputs import BaseVAE_Output

logger = logging.get_logger()


class LatentEncoderNTokens(nn.Module):
    '''
        Converts N hidden tokens into N seperate latent codes.
    '''
    def __init__(self, config):
        super().__init__()
        self.token_to_latent = nn.Linear(config.t5.d_model, config.latent_size)
        self.n_tokens = config.n_latent_tokens
        self.tanh = nn.Tanh()

    def forward(self, encoding) -> torch.Tensor:
        return self.tanh(self.token_to_latent(encoding))[:, : self.n_tokens, :]


class LatentEncoderNTokensSpectralFilter(nn.Module):
    '''
        Converts N hidden tokens with S spectral filters into N*S seperate latent codes.

    '''
    def __init__(self, config):
        super().__init__()
        self.n_tokens = config.n_latent_tokens
        self.tanh = nn.Tanh()
        self.spectral_filter_bands = config.spectral_filter_bands
        self.n_sectors = len(config.spectral_filter_bands)
        self.spectral_sector_size = config.t5.d_model // self.n_sectors
        self.token_to_latent = nn.Linear(self.spectral_sector_size, config.latent_size)
        self.spectral_coef = config.spectral_coef
        assert(config.t5.d_model % self.n_sectors == 0), 'Must have an equally sized band for each spectrum to make latent tokens.'

    def spectral_filter(self, H, low, high):
        H_dct = dct(H.T)
        H_dct[:, :low] *= self.spectral_coef
        H_dct[:, high:] *= self.spectral_coef
        return idct(H_dct).T

    def forward(self, encoding) -> torch.Tensor:
        # encoding (b, s, d)
        # -> (b, s, sectors, d/sectors)
        batch, seq_len = encoding.size()[:2]
        encoding = encoding.view(batch, seq_len, self.n_sectors, self.spectral_sector_size)
        for i, (low, high) in enumerate(self.spectral_filter_bands):
            encoding[:, :, i] = self.spectral_filter(encoding[:, :, i], low, high)
        return self.tanh(self.token_to_latent(encoding))[:, :self.n_tokens]


class LatentDecoderNTokens(nn.Module):
    '''
        Take several latent tokens and convert them each full token hidden states.
    '''
    def __init__(self, config):
        super().__init__()
        self.latent_size = config.latent_size
        if self.latent_size == config.t5.d_model:
            logger.warning('Latent decoder is not rescaling the latent code.')
            self.latent_to_token = lambda x: x
        else:
            self.latent_to_token = nn.Linear(self.latent_size, config.t5.d_model)
        self.norm = nn.LayerNorm(config.funnel.d_model, config.funnel.layer_norm_eps)

    def forward(self, latent) -> torch.Tensor:
        return self.norm(self.latent_to_token(latent))


class LatentDecoderNTokensSpectralFilter(LatentDecoderNTokens):
    '''
        Join spectrum tokens together.
    '''
    def __init__(self, config):
        config = copy.deepcopy(config)
        config.latent_size *= len(config.spectral_filter_bands)
        super().__init__(config)

    def forward(self, latent) -> torch.Tensor:
        # merge spectrum tokens
        return super().forward(latent.view(
            latent.size(0), latent.size(1), -1
        ))


VAE_ENCODER_MODELS = {
    "": LatentEncoderNTokens,
    "spectral": LatentEncoderNTokensSpectralFilter,
}
VAE_DECODER_MODELS = {
    "": LatentDecoderNTokens,
    "spectral": LatentDecoderNTokensSpectralFilter,
}


class EncoderDecoderVAE(nn.Module):
    """
    An MMD-VAE used with encoder-decoder models.
    Encodes all token encodings into a single latent & spits them back out.
    """

    batch_size = None

    def __init__(self, encoder, decoder, use_reg_loss=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.use_reg_loss = use_reg_loss

    def _model_forward(self, encoding, latent=None):
        if latent is None:
            latent = self.encoder(encoding)
        return self.decoder(latent), latent

    def forward(
        self,
        input_encoding=None,
        latent=None,
        skip_reg_loss=False,
    ):
        if input_encoding is None and latent is None:
            raise ValueError("Both `input_encoding` and `latent` sent to VAE are None.")
        use_reg_loss = self.use_reg_loss and latent is None and skip_reg_loss is False  # don't regularise if given latent
        recon_encoding, latent = self._model_forward(input_encoding, latent=latent)
        if use_reg_loss:
            # treat each token encoding as a seperate latent code
            latent_code_dim = latent.size()[-1]
            flat_latents = latent.reshape(-1, latent_code_dim)
            reg_loss = self._regularliser_loss(flat_latents) / flat_latents.size(0)
        else:
            reg_loss = torch.tensor(0, device=latent.device)
        return BaseVAE_Output(latent=latent, reconstructed_encoding=recon_encoding, reg_loss=reg_loss)

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

    def _regularliser_loss(self, latent):
        true_samples = torch.randn(latent.size(), device=latent.device)
        return self._compute_mmd(true_samples, latent)
