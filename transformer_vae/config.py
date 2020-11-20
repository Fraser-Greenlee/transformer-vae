import copy
from transformers.configuration_utils import PretrainedConfig
from transformers import (
    AutoConfig
)

from transformer_vae.autoencoders import VAE_ENCODER_MODELS, VAE_DECODER_MODELS


class Transformer_VAE_Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of :class:`~transformer_vae.T5_VAE_Model`.
    It is used to instantiate a T5-VAE model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration
    to that of the T5 `t5-vae-base architecture.

    To be able to use `transformer.trainer.Trainer` we need some specific training logic & config in the model.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Arguments:
        latent_size (:obj:`int`, `optional`, defaults to 1,000):
            Number of dimensions to use for the sequences latent code.
        transformer_name (:obj:`str`, `optional`, defaults to t5-base):
            Name of the transformer model to use as encoder & decoder.
        encoder_model (:obj:`str`, `optional`, defaults to None):
            Name of the model to encode T5 hidden states into latent codes.
        decoder_model (:obj:`str`, `optional`, defaults to None):
            Name of the model to decode latent codes into T5 hidden states.
        set_seq_size (:obj:`int`, `optional`, defaults to 60):
            NOTE: Every input sequence must be padded to be equal to this length.
        additional_latent_models (:obj:`list[nn.Module]`, `optional`, defaults to empty list):
            List of models that take the latent code and return a loss.
            Use this to condition the latent code on another model, optimising the latent space further.
        *** Training Args ***
        n_previous_latent_codes (:obj:`int`, `optional`, defaults to 3):
            Number of batches of previous latent codes to keep for MMD regularisation loss.
        reg_schedule_k (:obj:`float`, `optional`, defaults to 0.0025):
            Multiplied by global_step in a sigmoid, more gradually increase regulariser loss weight.
        reg_schedule_b (:obj:`float`, `optional`, defaults to 6.25):
            Added to global step in sigmoid, further delays increase in regulariser loss weight.
        use_extra_logs (:obj:`bool`, `optional`, defaults to False):
            Store extra logs during each training inference.
        *** End ***
    """
    model_type = "transformer_vae"
    is_composition = True

    def __init__(
        self,
        latent_size=1_000,
        transformer_name=None,
        encoder_model=None,
        decoder_model=None,
        set_seq_size=60,
        decoder_start_token_id=0,
        additional_latent_models=[],
        n_previous_latent_codes=3,
        reg_schedule_k=0.0025,
        reg_schedule_b=6.25,
        use_extra_logs=False,
        cache_dir=None,
        **kwargs,
    ):
        assert(encoder_model in VAE_ENCODER_MODELS)
        assert(decoder_model in VAE_DECODER_MODELS)
        super().__init__(**kwargs)
        self.transformer_config = AutoConfig.from_pretrained(transformer_name, cache_dir=cache_dir)
        self.transformer_config.n_positions = set_seq_size
        self.transformer_config.decoder_start_token_id = decoder_start_token_id
        self.latent_size = latent_size
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.set_seq_size = set_seq_size
        self.additional_latent_models = additional_latent_models
        self.n_previous_latent_codes = n_previous_latent_codes
        self.reg_schedule_k = reg_schedule_k
        self.reg_schedule_b = reg_schedule_b
        self.use_extra_logs = use_extra_logs

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default `to_dict()` from `PretrainedConfig`.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["transformer_config"] = self.transformer_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output


class T5_VAE_Config(Transformer_VAE_Config):
    def __init__(
        self,
        transformer_name='t5-base',
        **kwargs
    ):
        super().__init__(
            transformer_name=transformer_name,
            **kwargs
        )
        assert(self.transformer_config.model_type == 't5')


class Funnel_VAE_Config(Transformer_VAE_Config):
    r"""
    Arguments:
        encoded_seq_size (:obj:`int`, `optional`, defaults to 15):
            Size of the encoding sequence after all Funnel encoder blocks.
            Usually 1/4 of your input size.
    """
    def __init__(
        self,
        encoded_seq_size=60 // 4,
        transformer_name='funnel-transformer/large',
        **kwargs
    ):
        super().__init__(
            encoded_seq_size=encoded_seq_size,
            transformer_name=transformer_name,
            **kwargs
        )
        assert(self.transformer_config.model_type == 'funnel')


class Funnel_T5_VAE_Config(Transformer_VAE_Config):
    r"""
    Arguments:
        encoded_seq_size (:obj:`int`, `optional`, defaults to 15):
            Size of the encoding sequence after all Funnel encoder blocks.
            Usually 1/4 of your input size.
        transformer_decoder_name (:obj:`str`, `optional`, defaults to t5-base):
            Name of the Transformer model to use as encoder & decoder.
    """
    def __init__(
        self,
        encoded_seq_size=60 // 4,
        transformer_decoder_name='funnel-transformer/large',
        set_seq_size=60,
        decoder_start_token_id=0,
        cache_dir=None,
        **kwargs
    ):
        super().__init__(
            encoded_seq_size=encoded_seq_size,
            transformer_decoder_name=transformer_decoder_name,
            set_seq_size=set_seq_size,
            decoder_start_token_id=decoder_start_token_id,
            cache_dir=cache_dir,
            **kwargs
        )
        assert(self.transformer_config.model_type == 'funnel')
        self.transformer_decoder_config = AutoConfig.from_pretrained(transformer_decoder_name, cache_dir=cache_dir)
        self.transformer_decoder_config.n_positions = set_seq_size
        self.transformer_decoder_config.decoder_start_token_id = decoder_start_token_id

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default `to_dict()` from `PretrainedConfig`.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["transformer_config"] = self.transformer_config.to_dict()
        output["transformer_decoder_config"] = self.transformer_decoder_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
