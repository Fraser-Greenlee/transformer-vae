"""
    Define the T5-VAE model & all its variations.
"""
import pdb
import logging
import torch
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers import AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput

from t5_vae.autoencoders import VAE_ENCODER_MODELS, VAE_DECODER_MODELS
from t5_vae.modelling_outputs import BaseVAEOutput, VAE_Seq2SeqLMOutput
from t5_vae.config import T5_VAE_Config


logger = logging.getLogger(__name__)


class EncoderDecoderVAE(nn.Module):
    """
    An MMD-VAE used with encoder-decoder models.
    Encodes all token encodings into a single latent & spits them back out.
    """

    batch_size = None

    def __init__(self, encoder, decoder, use_n_previous_latent_codes):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.use_n_previous_latent_codes = use_n_previous_latent_codes
        self.prev_latents = None
        self.prev_latents_index = 0

    def _model_forward(self, encoding, latent=None):
        if latent is None:
            latent = self.encoder(encoding)
        return self.decoder(latent), latent

    def forward(
        self,
        input_encoding=None,
        latent_code=None,
    ):
        if input_encoding is None and latent_code is None:
            raise ValueError("Null input_encoding & latent_code sent to VAE.")
        recon_encoding, latent = self._model_forward(input_encoding, latent=latent_code)
        reg_loss = self._regularliser_loss(latent)
        return BaseVAEOutput(latent_code=latent, reconstructed_encoding=recon_encoding, reg_loss=reg_loss)

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

    def _get_combined_latents(self, latent):
        if self.prev_latents is None:
            # if no previous latents use this call to get the training batch size
            assert len(latent.size()) == 2
            self.batch_size = latent.size(0)
            self.prev_latents = torch.zeros((self.batch_size * self.use_n_previous_latent_codes, latent.size(1)), device=latent.device)
            # start by setting all previous to the first latent
            for i in range(self.use_n_previous_latent_codes):
                self.prev_latents[i * self.batch_size : (i + 1) * self.batch_size] = latent.detach()
        # update prev_latents to include new latents, overwriting the oldest ones
        return torch.cat((latent, self.prev_latents), 0)

    def _update_prev_latents(self, latent):
        if latent.size(0) < self.batch_size:
            logger.warn(
                f"Latent call has inconsistant batch size, skipping update previous latents. Expected: {self.batch_size} Got: {latent.size(0)}"
            )
            return None
        self.prev_latents[
            self.prev_latents_index * self.batch_size : (self.prev_latents_index + 1) * self.batch_size
        ] = latent.detach()
        self.prev_latents_index += 1
        if self.prev_latents_index >= self.use_n_previous_latent_codes:
            self.prev_latents_index = 0

    def _regularliser_loss(self, latent):
        if self.training:
            combined_latent = self._get_combined_latents(latent)
        else:
            combined_latent = latent
        true_samples = torch.randn(combined_latent.size()).to(combined_latent.device)
        result = self._compute_mmd(true_samples, combined_latent)
        if self.training:
            self._update_prev_latents(latent)
        return result


class T5_VAE_Model(PreTrainedModel):
    r"""
    The T5-VAE model was proposed in `Transformers as Variational Autoencoders
    <https://fraser-greenlee.github.io/2020/08/13/Transformers-as-Variational-Autoencoders.html>`__ by Fraser Greenlee.
    It is a modified T5 model that uses an MMD-VAE on sequence encodings to learn smooth latent spaces of discrete squences.

    NOTE: To work nicely with `huggingface.Trainer` this model handles lots some of its training logic here.
    - Must be trained with the `t5_vae.TellModelGlobalStep` for MMD regularising loss scheduling & log normalizing.
    - Must use `t5_vae.WandbCallbackUseModelLogs` for logging as it stores some of its own logs internally, using
      `get_latest_logs` to get the normalised logs and refresh the internal logs.

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc).

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~t5_vae.T5_VAE_Config`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
    """
    base_model_prefix = "t5_model"
    config_class = T5_VAE_Config
    global_step = None
    _calls_since_last_log = 0
    latest_logs = {
        "decoder_ce": 0,
        "reg_loss_w": 0,
        "reg_loss": 0,
    }
    _last_logs = {}

    def __init__(self, config: T5_VAE_Config):
        super().__init__(config=config)
        self.t5_model = AutoModelForSeq2SeqLM.from_config(config.t5_config)
        self.vae = EncoderDecoderVAE(
            VAE_ENCODER_MODELS[config.encoder_model](
                self.t5_model.config.d_model, self.config.set_seq_size, self.config.latent_size
            ),
            VAE_DECODER_MODELS[config.decoder_model](
                self.t5_model.config.d_model, self.config.set_seq_size, self.config.latent_size, self.t5_model.config
            ),
            self.config.n_previous_latent_codes,
        )

    def resize_token_embeddings(self, *args, **kwargs):
        super().resize_token_embeddings(*args, **kwargs)
        self.t5_model.resize_token_embeddings(*args, **kwargs)

    def get_input_embeddings(self):
        return self.t5_model.shared

    def set_input_embeddings(self, new_embeddings):
        return self.t5_model.set_input_embeddings(new_embeddings)

    def _init_weights(self, module):
        return self.t5_model._init_weights(module)

    def decoder_logits(self, decoder_input_ids, encoding):
        sequence_output = self.t5_model.decoder(input_ids=decoder_input_ids, encoder_hidden_states=encoding)[0]
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.t5_model.model_dim ** -0.5)
        logits = self.t5_model.lm_head(sequence_output)
        return logits

    def _regulariser_loss_weight_schedule(self):
        if self.global_step is None:
            return 1
        return torch.sigmoid(
            torch.tensor(self.global_step * self.config.reg_schedule_k - self.config.reg_schedule_b)
        ).item()

    def _update_logs(self, **logs):
        self._calls_since_last_log += 1
        for k, v in logs.items():
            self.latest_logs[k] = self.latest_logs.get(k, 0) + v

    def get_latest_logs(self):
        """
            Gets latest logs and refreshes the log values.

            Logs are normalised by the number of training inferences since the last log.
        """
        assert self.config.use_extra_logs

        result = dict(self.latest_logs)
        for k, v in result.items():
            value_increase = (v - self._last_logs.get(k, 0))
            result[k] = value_increase / self._calls_since_last_log

        self._last_logs = dict(self.latest_logs)
        self._calls_since_last_log = 0

        return result

    def forward(
        self,
        input_ids=None,
        labels=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        latent_code=None,
    ):
        if input_ids is not None:
            if attention_mask is None:
                attention_mask = input_ids.ne(self.t5_model.config.pad_token_id).long()
            if encoder_outputs is None:
                encoder_outputs = self.t5_model.encoder(
                    input_ids=input_ids, attention_mask=attention_mask, return_dict=True
                )
        if encoder_outputs is not None and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        vae_outputs = self.vae(
            input_encoding=encoder_outputs.last_hidden_state if encoder_outputs else None, latent_code=latent_code
        )

        if labels is not None and decoder_input_ids is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self.t5_model._shift_right(labels) if labels is not None else None

        decoder_outputs = self.t5_model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=vae_outputs.reconstructed_encoding,
        )

        sequence_output = decoder_outputs[0]
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.t5_model.model_dim ** -0.5)
        lm_logits = self.t5_model.lm_head(sequence_output)

        decoder_ce = torch.tensor(0.0, device=lm_logits.device)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            decoder_ce = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        reg_loss_w = self._regulariser_loss_weight_schedule()
        loss = decoder_ce + vae_outputs.reg_loss * reg_loss_w

        if reg_loss_w > 1.0:
            pdb.set_trace()

        if self.training and self.config.use_extra_logs:
            self._update_logs(decoder_ce=decoder_ce.item(), reg_loss=vae_outputs.reg_loss.item(), reg_loss_w=reg_loss_w)

        return VAE_Seq2SeqLMOutput(
            reg_loss=vae_outputs.reg_loss,
            decoder_ce=decoder_ce,
            reconstructed_encoding=vae_outputs.reconstructed_encoding,
            loss=loss,
            latnet=vae_outputs.latent_code,
        )
