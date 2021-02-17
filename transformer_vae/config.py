import copy
from logging import warning
import math
from transformers.utils import logging
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoConfig

from transformer_vae.autoencoders import VAE_ENCODER_MODELS, VAE_DECODER_MODELS
from transformer_vae.utils import assertEqual, assertIn

logger = logging.get_logger(__name__)


def _test_overlap(s, w, o):
    window_max = 0
    while window_max < s:
        start = max(window_max - o, 0)
        end = min(w, s - start)
        if window_max == 0:
            window_max += w
        else:
            window_max += w - o
    return w - end


class Funnel_T5_VAE_Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of :class:`~transformer_vae.T5_VAE_Model`.
    It is used to instantiate a Funnel-T5-VAE model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the T5 `funnel-t5-vae-base architecture.

    To be able to use `transformer.trainer.Trainer` we need some specific training logic & config in the model.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Arguments:
        latent_size (:obj:`int`, `optional`, defaults to 1,000):
            Number of dimensions to use for the sequences latent code.
        funnel_name (:obj:`str`, `optional`, defaults to t5-base):
            Name of the transformer model to use as encoder & decoder.
        vae_encoder_model (:obj:`str`, `optional`, defaults to None):
            Name of the model to encode T5 hidden states into latent codes.
        vae_decoder_model (:obj:`str`, `optional`, defaults to None):
            Name of the model to decode latent codes into T5 hidden states.
        set_seq_size (:obj:`int`, `optional`, defaults to 60):
            NOTE: Every input sequence must be padded to be equal to this length.
        t5_name (:obj:`str`, `optional`, defaults to t5-base):
            Name of the Transformer model to use as a decoder.
        transformer_critic_name (:obj:`str`, `optional`, defaults to None):
            Name of the Transformer model to use as an advisery on interpolations.
        *** Training Args ***
        reg_schedule_k (:obj:`float`, `optional`, defaults to 0.0025):
            Multiplied by global_step in a sigmoid, more gradually increase regulariser loss weight.
        reg_schedule_b (:obj:`float`, `optional`, defaults to 6.25):
            Added to global step in sigmoid, further delays increase in regulariser loss weight.
        use_extra_logs (:obj:`bool`, `optional`, defaults to False):
            Store extra logs during each training inference.
        gradient_checkpoint (:obj:`bool`, `optional`, defaults to False):
            Checkpoint gradients in the model.
            Currently just checkpoints after the encoder + VAE
        funnel_block_sizes (:obj:`str`, defaults to '1_1_1'):
            Size of each Funnel Encoder block, sequence is halved between each block.
        *** End ***

        TODO: Add extra models to condition on the latent
    """
    model_type = "transformer_vae"
    is_composition = True

    def __init__(
        self,
        latent_size=1_000,
        funnel_name="funnel-transformer/intermediate",
        t5_name="t5-base",
        vae_encoder_model=None,
        vae_decoder_model=None,
        critic_type=None,
        critic_name=None,
        set_seq_size=60,
        decoder_start_token_id=0,
        use_reg_loss=True,
        reg_schedule_k=0.0025,
        reg_schedule_b=6.25,
        use_extra_logs=False,
        cache_dir=None,
        n_latent_tokens=5,  # set to -1 for full sequence
        funnel_block_sizes='1_1_1',
        attention_window_size=0,
        attention_window_overlap=0,
        gradient_checkpoint_encoder=False,
        decoder_grad_accumulation_rate=0,
        skip_upsample=False,
        **kwargs,
    ):
        assertIn(vae_encoder_model, VAE_ENCODER_MODELS.keys(), "Unexpected VAE encoder.")
        assertIn(vae_decoder_model, VAE_DECODER_MODELS.keys(), "Unexpected VAE decoder.")

        super().__init__(**kwargs)

        # VAE
        self.vae_encoder_model = vae_encoder_model
        self.vae_decoder_model = vae_decoder_model
        if set_seq_size < n_latent_tokens:
            logger.warning(f'set_seq_size size is smaller than n_latent_tokens, now using n_latent_tokens={set_seq_size} from {n_latent_tokens}')
            n_latent_tokens = set_seq_size
        self.latent_size = latent_size
        self.n_latent_tokens = n_latent_tokens
        self.skip_upsample = skip_upsample

        # funnel encoder model
        self.funnel = AutoConfig.from_pretrained(funnel_name, cache_dir=cache_dir)
        self.funnel.block_sizes = [int(i) for i in funnel_block_sizes.split('_')]
        self.funnel.decoder_start_token_id = decoder_start_token_id
        self.funnel.n_positions = set_seq_size
        pooling_division = 2 ** (len(self.funnel.block_sizes) - 1)
        self.encoded_seq_size = math.ceil(self.funnel.n_positions / pooling_division)
        self.gradient_checkpoint_encoder = gradient_checkpoint_encoder

        # T5 decoder model
        self.t5 = AutoConfig.from_pretrained(t5_name, cache_dir=cache_dir)
        self.t5.decoder_start_token_id = decoder_start_token_id
        self.t5.n_positions = self.funnel.n_positions
        assertEqual(self.t5.model_type, "t5", "Need t5 model type for transformer_decoder.")
        assertEqual(self.funnel.d_model, self.t5.d_model, "Funnel & T5 transformers have different dimensions.")
        self.decoder_grad_accumulation_rate = decoder_grad_accumulation_rate
        assert(attention_window_size < set_seq_size), 'Attention window must be smallar than set sequence size.'
        self.attention_window_size = attention_window_size
        self.attention_window_overlap = attention_window_overlap
        if attention_window_size:
            assert(set_seq_size % attention_window_size != 0), 'When doing an alternating attention pattern the sequence size cannot be divisable by the window size as no alternations will be possible.'
            self.attention_window_overlap = set_seq_size % attention_window_size

        # extra training losses
        self.use_reg_loss = use_reg_loss
        if not use_reg_loss:
            logger.warning("Regularisation loss is turned off, you are training an Autoencoder (not a VAE).")
        self.reg_schedule_k = reg_schedule_k
        self.reg_schedule_b = reg_schedule_b
        self.use_extra_logs = use_extra_logs

        # critic model
        self.critic = None
        if critic_name:
            self.critic_type = critic_type
            self.critic = AutoConfig.from_pretrained(critic_name, cache_dir=cache_dir)
            assertEqual(self.t5.d_model, self.critic.d_model, "Funnel & T5 transformers have different dimensions.")

        # misc
        self.use_cache = getattr(self.funnel, "use_cache", False)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default `to_dict()` from `PretrainedConfig`.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["funnel"] = self.funnel.to_dict()
        output["model_type"] = self.__class__.model_type
        output['t5'] = self.t5.to_dict()
        if self.critic:
            output['critic'] = self.critic.to_dict()
        return output
