"""
    Base transformer-VAE model.
"""
import torch
from torch import nn
from typing import Dict, Any
from transformers.utils import logging
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.funnel.modeling_funnel import upsample
from transformers import AutoModelForSeq2SeqLM, AutoModelForMaskedLM

from transformer_vae.custom_t5 import modify_t5_stack
from transformer_vae.autoencoders import VAE_ENCODER_MODELS, VAE_DECODER_MODELS, EncoderDecoderVAE
from transformer_vae.critic import CRITIC
from transformer_vae.model_outputs import BaseTransformerVAE_Output
from transformer_vae.config import Funnel_T5_VAE_Config


logger = logging.get_logger(__name__)


class Funnel_T5_VAE_Model(PreTrainedModel):
    r"""
    The Funnel-VAE model was proposed in `Transformers as Variational Autoencoders
    <https://fraser-greenlee.github.io/2020/08/13/Transformers-as-Variational-Autoencoders.html>`__ by Fraser Greenlee.
    It is a modified Funnel-Transformer model that uses an MMD-VAE on sequence encodings to learn smooth latent spaces of discrete squences.

    Funnel-VAE has its input sequence compressed & then upsampled by Funnel-Transformer.
    This makes it better able to model long sequences.
    Funnel-Transformer's decoder is non auto-regressive meaning it generates all tokens in parallel, this is likely worse for generation.

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
        config (:class:`~transformer_vae.Funnel_T5_VAE_Config`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
    """
    config_class = Funnel_T5_VAE_Config
    base_model_prefix = "transformer"
    global_step = None
    _calls_since_last_log = 0
    latest_logs = {
        "decoder_ce": 0,
        "seq_accuracy": 0,
        "token_accuracy": 0,
        "reg_loss_w": 0,
        "reg_loss": 0,
        'critic_loss_on_model': 0,
        'critic_loss': 0,
    }
    _last_logs: Dict[str, float] = {}

    def __init__(self, config: Funnel_T5_VAE_Config):
        super().__init__(config=config)
        funnel_transformer = AutoModelForMaskedLM.from_config(config.funnel)
        t5_transformer = AutoModelForSeq2SeqLM.from_config(config.t5)

        self.encoder = funnel_transformer.funnel.encoder
        self.decoder = modify_t5_stack(t5_transformer.decoder, config)
        self.lm_head = t5_transformer.lm_head
        self.shared_embedding = t5_transformer.shared
        self.decoder_start_token_id = self.config.t5.decoder_start_token_id
        assert (
            self.decoder_start_token_id is not None
        ), "`self.config.t5.decoder_start_token_id` has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

        self.vae = EncoderDecoderVAE(
            VAE_ENCODER_MODELS[config.vae_encoder_model](self.config),
            VAE_DECODER_MODELS[config.vae_decoder_model](self.config),
            self.config.use_reg_loss,
        )

        self.critic = None
        if config.critic:
            self.critic = CRITIC[config.critic_type](config.critic)

    def get_input_embeddings(self):
        return self.shared_embedding

    def set_input_embeddings(self, new_embeddings):
        self.shared_embedding = new_embeddings

    def _init_weights(self, module):
        pass

    def _regulariser_loss_weight_schedule(self):
        if self.global_step is None or not self.config.use_reg_loss:
            return 0
        # edit using https://www.desmos.com/calculator/mqzxhecfxz
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

    def _get_encoder_outputs(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        inputs_embeds=None,
        return_dict=True,
        # unused args
        class_label=None,
        label=None,
    ):
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

        if inputs_embeds is None:
            inputs_embeds = self.shared_embedding(input_ids)

        if self.config.gradient_checkpoint_encoder:

            def create_custom_forward(encoder):
                def custom_forward(*inputs):
                    return encoder(*inputs, False, False, False)
                return custom_forward

            return torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.encoder),
                inputs_embeds,
                attention_mask,
                token_type_ids,
            )

        return self.encoder(
            inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=False,
            return_dict=True,
        )

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.t5.decoder_start_token_id
        pad_token_id = self.config.t5.pad_token_id

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
        inputs_embeds=None,
        labels=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        latent=None,
        use_cache=None,
        output_hidden_states=None,
        return_dict=True,
        **unused_kwargs
    ):
        assert return_dict, "Need return_dict=True, using tuple's is not implimented"
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is not None or inputs_embeds is not None:
            if decoder_input_ids is not None and input_ids.equal(decoder_input_ids) is False:
                raise ValueError(
                    "`input_ids` and `decoder_input_ids` do not match. Funnel-T5-VAE can only reproduce its input sequence."
                )
            if attention_mask is None and input_ids is not None:
                attention_mask = input_ids.ne(self.config.t5.pad_token_id).long()
            if encoder_outputs is None:
                encoder_outputs = self._get_encoder_outputs(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
        if encoder_outputs is not None and (isinstance(encoder_outputs, list) or isinstance(encoder_outputs, tuple)):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        vae_outputs = self.vae(
            input_encoding=encoder_outputs.last_hidden_state if encoder_outputs and isinstance(encoder_outputs, BaseModelOutput) else None, latent=latent
        )

        if self.config.skip_upsample:
            upsampled_encoding = vae_outputs.reconstructed_encoding
        else:
            upsampled_encoding = upsample(
                vae_outputs.reconstructed_encoding,
                stride=2 ** (len(self.config.funnel.block_sizes) - 1),
                target_len=self.config.t5.n_positions,
                separate_cls=self.config.funnel.separate_cls,
                truncate_seq=self.config.funnel.truncate_seq,
            )

        # Now using T5 decoder

        if labels is not None and decoder_input_ids is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels) if labels is not None else None

        decoder_ce = torch.tensor(0.0, device=upsampled_encoding.device)
        seq_accuracy = torch.tensor(0.0, device=upsampled_encoding.device)
        token_accuracy = torch.tensor(0.0, device=upsampled_encoding.device)
        decoder_outputs = None
        lm_logits = None
        if decoder_input_ids is not None:
            # , grad_chk_pnt_rate=self.config.decoder_grad_chk_pnt_rate
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids, encoder_hidden_states=upsampled_encoding, use_cache=use_cache, output_hidden_states=output_hidden_states, return_dict=True
            )

            sequence_output = decoder_outputs.last_hidden_state
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.config.t5.d_model ** -0.5)
            lm_logits = self.lm_head(sequence_output)

            if labels is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                decoder_ce = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
                chosen_tokens = torch.argmax(lm_logits, 2)
                pad_tokens = (labels == -100).int()
                correct_tokens = (chosen_tokens == labels).int() + pad_tokens
                seq_accuracy = (torch.min(correct_tokens, dim=1).values.sum() / labels.size(0)).detach()
                num_pad_tokens = pad_tokens.sum()
                token_accuracy = ((correct_tokens.sum() - num_pad_tokens) / (labels.numel() - num_pad_tokens)).detach()

        reg_loss_w = self._regulariser_loss_weight_schedule()
        loss = decoder_ce + vae_outputs.reg_loss * reg_loss_w

        if self.training and self.config.use_extra_logs:
            self._update_logs(
                decoder_ce=decoder_ce.item(), seq_accuracy=seq_accuracy, token_accuracy=token_accuracy, reg_loss=vae_outputs.reg_loss.item(), reg_loss_w=reg_loss_w
            )

        return BaseTransformerVAE_Output(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values if decoder_outputs else None,
            decoder_hidden_states=decoder_outputs.hidden_states if decoder_outputs else None,
            hidden_states=decoder_outputs.hidden_states if decoder_outputs else None,
            decoder_attentions=decoder_outputs.attentions if decoder_outputs else None,
            cross_attentions=decoder_outputs.cross_attentions if decoder_outputs else None,
            reconstructed_encoding=vae_outputs.reconstructed_encoding,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state if encoder_outputs else None,
            encoder_hidden_states=encoder_outputs.hidden_states if encoder_outputs else None,
            encoder_attentions=encoder_outputs.attentions if encoder_outputs else None,
            latent=vae_outputs.latent,
            reg_loss=vae_outputs.reg_loss,
            decoder_ce=decoder_ce,
            seq_accuracy=seq_accuracy,
            token_accuracy=token_accuracy
        )
