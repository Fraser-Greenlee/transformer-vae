"""
    Base transformer-VAE model.
"""
import logging
import torch
from torch import nn
from typing import Dict, Any
from transformers.modeling_utils import PreTrainedModel
from transformers import AutoModelForSeq2SeqLM, AutoModelForMaskedLM
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.funnel.modeling_funnel import upsample

from transformer_vae.autoencoders import VAE_ENCODER_MODELS, VAE_DECODER_MODELS
from transformer_vae.model_outputs import BaseVAE_Output, BaseTransformerVAE_Output
from transformer_vae.config import Transformer_VAE_Config

from transformer_vae.config import T5_VAE_Config, Funnel_VAE_Config, Funnel_T5_VAE_Config


logger = logging.getLogger(__name__)


class EncoderDecoderVAE(nn.Module):
    """
    An MMD-VAE used with encoder-decoder models.
    Encodes all token encodings into a single latent & spits them back out.
    """

    batch_size = None

    def __init__(self, encoder, decoder, use_n_previous_latent_codes=0, smaller_mmd_batch_size=None, use_reg_loss=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.use_n_previous_latent_codes = use_n_previous_latent_codes
        self.smaller_mmd_batch_size = smaller_mmd_batch_size
        if smaller_mmd_batch_size:
            assert use_n_previous_latent_codes == 0, "Can't use smaller mmd batch size AND use previous latent codes."
        self.prev_latents = None
        self.prev_latents_index = 0
        self.use_reg_loss = use_reg_loss

    def _model_forward(self, encoding, latent=None):
        if latent is None:
            latent = self.encoder(encoding)
        return self.decoder(latent), latent

    def forward(
        self,
        input_encoding=None,
        latent=None,
    ):
        if input_encoding is None and latent is None:
            raise ValueError("Both `input_encoding` and `latent` sent to VAE are Null.")
        recon_encoding, latent = self._model_forward(input_encoding, latent=latent)
        reg_loss = torch.tensor(0, device=latent.device)
        if self.use_reg_loss:
            reg_loss = self._regularliser_loss(latent)
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

    def _get_combined_latents(self, latent):
        if self.prev_latents is None:
            # if no previous latents use this call to get the training batch size
            assert len(latent.size()) == 2
            self.batch_size = latent.size(0)
            self.prev_latents = torch.zeros(
                (self.batch_size * self.use_n_previous_latent_codes, latent.size(1)), device=latent.device
            )
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

    def _using_prev_latents(self):
        return self.training and self.use_n_previous_latent_codes > 0

    def _regularliser_loss(self, latent):
        if self.training and self.smaller_mmd_batch_size:
            batch_size = latent.size(0)
            if batch_size // self.smaller_mmd_batch_size != batch_size / self.smaller_mmd_batch_size:
                return self._batch_of_regularliser_loss(latent)
            all_latents = latent.view(
                latent.size(0) // self.smaller_mmd_batch_size, self.smaller_mmd_batch_size, latent.size(1)
            )
            total = torch.tensor(0.0, device=latent.device)
            for latent_batch in all_latents:
                total += self._batch_of_regularliser_loss(latent_batch)
            return total
        return self._batch_of_regularliser_loss(latent)

    def _batch_of_regularliser_loss(self, latent):
        if self._using_prev_latents():
            combined_latent = self._get_combined_latents(latent)
        else:
            combined_latent = latent
        true_samples = torch.randn(combined_latent.size()).to(combined_latent.device)
        result = self._compute_mmd(true_samples, combined_latent)
        if self._using_prev_latents():
            self._update_prev_latents(latent)
        return result


class Transformer_VAE_Base_Model(PreTrainedModel):
    r"""
    This is the base for Transformer-VAE's.
    Each Transformer-VAE takes an encoder-decoder transformer and converts the encoder's encoding into a latent code & back into an encoding again.
    These latent codes can then be modified to produce new encodings and so new sequences.

    NOTE: To work nicely with `huggingface.Trainer` this model handles some of its training logic here.
    - Must be trained with the `transformer_vae.TellModelGlobalStep` for MMD regularising loss scheduling & log normalizing.
    - Must use `transformer_vae.WandbCallbackUseModelLogs` for logging as it stores some of its own logs internally, using
      `get_latest_logs` to get the normalised logs and refresh the internal logs.

    NOTE: Its generation works differently. Instead of taking input_ids and sampling form the decoder it takes a `latent`
    and uses `input_ids` as `decoder_input_ids`.

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc).

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformer_vae.Transformer_VAE_Config`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
    """
    base_model_prefix = "transformer"
    # config_class # impliment this!
    global_step = None
    _calls_since_last_log = 0
    latest_logs = {
        "decoder_ce": 0,
        "reg_loss_w": 0,
        "reg_loss": 0,
    }
    _last_logs: Dict[str, float] = {}

    def __init__(self, config: Transformer_VAE_Config):
        super().__init__(config=config)

        if config.transformer.model_type == "t5":
            self.transformer = AutoModelForSeq2SeqLM.from_config(config.transformer)
        elif config.transformer.model_type == "funnel":
            self.transformer = AutoModelForMaskedLM.from_config(config.transformer)
        else:
            raise ValueError(f'Unrecognised model type: "{config.transformer.model_type }"')

        self.vae = EncoderDecoderVAE(
            VAE_ENCODER_MODELS[config.encoder_model](
                self.transformer.config.d_model, self.config.encoded_seq_size, self.config.latent_size
            ),
            VAE_DECODER_MODELS[config.decoder_model](
                self.transformer.config.d_model,
                self.config.encoded_seq_size,
                self.config.latent_size,
                self.transformer.config,
            ),
            self.config.n_previous_latent_codes,
            self.config.mmd_batch_size,
            self.config.use_reg_loss,
        )

    def resize_token_embeddings(self, *args, **kwargs):
        super().resize_token_embeddings(*args, **kwargs)
        self.transformer.resize_token_embeddings(*args, **kwargs)

    def get_input_embeddings(self):
        return self.transformer.shared

    def set_input_embeddings(self, new_embeddings):
        return self.transformer.set_input_embeddings(new_embeddings)

    def _init_weights(self, module):
        return self.transformer._init_weights(module)

    def _regulariser_loss_weight_schedule(self):
        if self.global_step is None or not self.config.use_reg_loss:
            return 0
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
        if self._calls_since_last_log < 1:
            return {}

        result = dict(self.latest_logs)
        for k, v in result.items():
            value_increase = v - self._last_logs.get(k, 0)
            result[k] = value_increase / self._calls_since_last_log

        self._last_logs = dict(self.latest_logs)
        self._calls_since_last_log = 0

        return result

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, latent=None, **kwargs) -> Dict[str, Any]:
        """
        Should only be generating text from latent codes.
        """
        assert (
            latent is not None
        ), "Generation with Transformer-VAE's expects to be given a latent code to generate from."
        for rm_key in ["past", "attention_mask"]:
            if rm_key in kwargs:
                del kwargs[rm_key]
        return {"decoder_input_ids": input_ids, "latent": latent, **kwargs}

    def forward(
        self,
        input_ids=None,
        labels=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        latent=None,
        return_dict=True,
        class_label=None,
    ):
        raise NotImplementedError()


class T5_VAE_Model(Transformer_VAE_Base_Model):
    r"""
    The T5-VAE model was proposed in `Transformers as Variational Autoencoders
    <https://fraser-greenlee.github.io/2020/08/13/Transformers-as-Variational-Autoencoders.html>`__ by Fraser Greenlee.
    It is a modified T5 model that uses an MMD-VAE on sequence encodings to learn smooth latent spaces of discrete squences.

    T5-VAE only compresses its encodings after the encoder with a few fully connected layers making it less effective at modelling long sequences.
    Its decoder is autoregressive making it natually effective at generating sequences.
    """
    config_class = T5_VAE_Config

    def _shift_input_right(self, input_ids):
        start_token_id = self.transformer.config.eos_token_id
        pad_token_id = self.config.transformer_decoder.pad_token_id

        assert (
            start_token_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"
        assert start_token_id != pad_token_id, "Trying to prepend the padding token to the sequence."

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids

    def forward(
        self,
        input_ids=None,
        labels=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        latent=None,
        use_cache=None,
        return_dict=True,
        class_label=None,
    ):
        assert return_dict, "Need return_dict=True, using tuple's is not implimented"
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is not None:
            if self.config.prepend_eos_token:
                input_ids = self._shift_input_right(input_ids)
            if attention_mask is None:
                attention_mask = input_ids.ne(self.transformer.config.pad_token_id).long()
            if encoder_outputs is None:
                encoder_outputs = self.transformer.encoder(
                    input_ids=input_ids, attention_mask=attention_mask, return_dict=True
                )
        if encoder_outputs is not None and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        vae_outputs = self.vae(
            input_encoding=encoder_outputs.last_hidden_state if encoder_outputs else None, latent=latent
        )

        if labels is not None and decoder_input_ids is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self.transformer._shift_right(labels) if labels is not None else None

        decoder_outputs = self.transformer.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=vae_outputs.reconstructed_encoding,
            use_cache=use_cache,
            return_dict=True,
        )

        sequence_output = decoder_outputs.last_hidden_state
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.config.transformer.d_model ** -0.5)
        lm_logits = self.transformer.lm_head(sequence_output)

        decoder_ce = torch.tensor(0.0, device=lm_logits.device)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            decoder_ce = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        reg_loss_w = self._regulariser_loss_weight_schedule()
        loss = decoder_ce + vae_outputs.reg_loss * reg_loss_w

        if self.training and self.config.use_extra_logs:
            self._update_logs(decoder_ce=decoder_ce.item(), reg_loss=vae_outputs.reg_loss.item(), reg_loss_w=reg_loss_w)

        return BaseTransformerVAE_Output(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state if encoder_outputs else None,
            encoder_hidden_states=encoder_outputs.hidden_states if encoder_outputs else None,
            encoder_attentions=encoder_outputs.attentions if encoder_outputs else None,
            latent=vae_outputs.latent,
            reg_loss=vae_outputs.reg_loss,
            decoder_ce=decoder_ce,
        )


class Funnel_VAE_Model_Base(Transformer_VAE_Base_Model):
    def _get_encoder_outputs(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        class_label=None,
    ):
        funnel = self.transformer.funnel

        output_attentions = output_attentions if output_attentions is not None else funnel.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else funnel.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else funnel.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both `input_ids` and `inputs_embeds` at the same time.")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either `input_ids` or `inputs_embeds`")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # TODO: deal with head_mask
        if inputs_embeds is None:
            inputs_embeds = funnel.embeddings(input_ids)

        return funnel.encoder(
            inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )


class Funnel_VAE_Model(Funnel_VAE_Model_Base):
    r"""
    The Funnel-VAE-T5 model was proposed in `Transformers as Variational Autoencoders
    <https://fraser-greenlee.github.io/2020/08/13/Transformers-as-Variational-Autoencoders.html>`__ by Fraser Greenlee.
    It is a modified Funnel-Transformer model that uses an MMD-VAE on its sequence encodings and a T5 decoder.

    Funnel-VAE has its input sequence compressed & then upsampled by Funnel-Transformer.
    This makes it better able to model long sequences.
    Its decoder is autoregressive making it natually effective at generating sequences.
    """
    config_class = Funnel_VAE_Config

    def forward(
        self,
        input_ids=None,
        labels=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        latent=None,
        use_cache=None,
        return_dict=True,
        class_label=None,
    ):
        assert return_dict, "Need return_dict=True, using tuple's is not implimented"

        if input_ids is not None:
            if decoder_input_ids is not None and input_ids.equal(decoder_input_ids) is False:
                raise ValueError(
                    "`input_ids` and `decoder_input_ids` do not match. Funnel-VAE can only reproduce its input sequence."
                )
            if self.config.prepend_eos_token:
                raise NotImplementedError()
            if attention_mask is None:
                attention_mask = input_ids.ne(self.transformer.config.pad_token_id).long()
            if encoder_outputs is None:
                encoder_outputs = self._get_encoder_outputs(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
        if encoder_outputs is not None and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        vae_outputs = self.vae(
            input_encoding=encoder_outputs.last_hidden_state if encoder_outputs else None, latent=latent
        )

        initial_encoding_size = (
            vae_outputs.reconstructed_encoding.size(0),
            self.config.transformer.n_positions,
            self.config.transformer.d_model,
        )

        decoder_outputs = self.transformer.funnel.decoder(
            final_hidden=vae_outputs.reconstructed_encoding,
            # Don't allow for residual connections, instead just send an empty tensor.
            first_block_hidden=torch.zeros(initial_encoding_size, device=vae_outputs.reconstructed_encoding.device),
            return_dict=True,
        )

        last_hidden_state = decoder_outputs.last_hidden_state
        prediction_logits = self.transformer.lm_head(last_hidden_state)

        decoder_ce = torch.tensor(0.0, device=prediction_logits.device)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            decoder_ce = loss_fct(prediction_logits.view(-1, self.config.transformer.vocab_size), labels.view(-1))

        reg_loss_w = self._regulariser_loss_weight_schedule()
        loss = decoder_ce + vae_outputs.reg_loss * reg_loss_w

        if self.training and self.config.use_extra_logs:
            self._update_logs(decoder_ce=decoder_ce.item(), reg_loss=vae_outputs.reg_loss.item(), reg_loss_w=reg_loss_w)

        return BaseTransformerVAE_Output(
            loss=loss,
            logits=prediction_logits,
            past_key_values=None,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=None,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state if encoder_outputs else None,
            encoder_hidden_states=encoder_outputs.hidden_states if encoder_outputs else None,
            encoder_attentions=encoder_outputs.attentions if encoder_outputs else None,
            latent=vae_outputs.latent,
            reg_loss=vae_outputs.reg_loss,
            decoder_ce=decoder_ce,
        )


class Funnel_T5_VAE_Model(Funnel_VAE_Model_Base):
    r"""
    The Funnel-VAE model was proposed in `Transformers as Variational Autoencoders
    <https://fraser-greenlee.github.io/2020/08/13/Transformers-as-Variational-Autoencoders.html>`__ by Fraser Greenlee.
    It is a modified Funnel-Transformer model that uses an MMD-VAE on sequence encodings to learn smooth latent spaces of discrete squences.

    Funnel-VAE has its input sequence compressed & then upsampled by Funnel-Transformer.
    This makes it better able to model long sequences.
    Funnel-Transformer's decoder is non auto-regressive meaning it generates all tokens in parallel, this is likely worse for generation.
    """
    config_class = Funnel_T5_VAE_Config

    def __init__(self, config: Funnel_T5_VAE_Config):
        super().__init__(config=config)
        t5_model = AutoModelForSeq2SeqLM.from_config(config.transformer_decoder)
        self.transformer.decoder = t5_model.decoder
        self.transformer.lm_head = t5_model.lm_head

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.transformer_decoder.decoder_start_token_id
        pad_token_id = self.config.transformer_decoder.pad_token_id

        assert (
            decoder_start_token_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids

    def forward(
        self,
        input_ids=None,
        labels=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        latent=None,
        use_cache=None,
        return_dict=True,
        class_label=None,
    ):
        assert return_dict, "Need return_dict=True, using tuple's is not implimented"
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is not None:
            if decoder_input_ids is not None and input_ids.equal(decoder_input_ids) is False:
                raise ValueError(
                    "`input_ids` and `decoder_input_ids` do not match. Funnel-VAE can only reproduce its input sequence."
                )
            if self.config.prepend_eos_token:
                raise NotImplementedError()
            if attention_mask is None:
                attention_mask = input_ids.ne(self.transformer.config.pad_token_id).long()
            if encoder_outputs is None:
                encoder_outputs = self._get_encoder_outputs(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
        if encoder_outputs is not None and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        vae_outputs = self.vae(
            input_encoding=encoder_outputs.last_hidden_state if encoder_outputs else None, latent=latent
        )

        # TODO allow more options here
        if self.config.padding_input:
            upsampled_encoding = upsample(
                vae_outputs.reconstructed_encoding,
                stride=2 ** (len(self.config.transformer.block_sizes) - 1),
                target_len=self.config.transformer_decoder.n_positions,
                separate_cls=self.config.transformer.separate_cls,
                truncate_seq=self.config.transformer.truncate_seq,
            )
        else:
            upsampled_encoding = vae_outputs.reconstructed_encoding

        # Now using T5 decoder

        if labels is not None and decoder_input_ids is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels) if labels is not None else None

        decoder_outputs = self.transformer.decoder(
            input_ids=decoder_input_ids, encoder_hidden_states=upsampled_encoding, use_cache=use_cache, return_dict=True
        )

        sequence_output = decoder_outputs.last_hidden_state
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.config.transformer.d_model ** -0.5)
        lm_logits = self.transformer.lm_head(sequence_output)

        decoder_ce = torch.tensor(0.0, device=lm_logits.device)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            decoder_ce = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        reg_loss_w = self._regulariser_loss_weight_schedule()
        loss = decoder_ce + vae_outputs.reg_loss * reg_loss_w

        if self.training and self.config.use_extra_logs:
            self._update_logs(decoder_ce=decoder_ce.item(), reg_loss=vae_outputs.reg_loss.item(), reg_loss_w=reg_loss_w)

        return BaseTransformerVAE_Output(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state if encoder_outputs else None,
            encoder_hidden_states=encoder_outputs.hidden_states if encoder_outputs else None,
            encoder_attentions=encoder_outputs.attentions if encoder_outputs else None,
            latent=vae_outputs.latent,
            reg_loss=vae_outputs.reg_loss,
            decoder_ce=decoder_ce,
        )
