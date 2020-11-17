import copy
from datasets.load import import_main_class
from transformers.configuration_utils import PretrainedConfig
from transformers.configuration_t5 import T5Config


class T5_VAE_Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of :class:`~t5_vae.T5_VAE_Model`.
    It is used to instantiate a T5-VAE model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration
    to that of the T5 `t5-vae-base architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Arguments:
        t5_model_name (:obj:`str`, `optional`, defaults to t5-base):
            Name of the T5 model to use as encoder & decoder.
        latent_size (:obj:`int`, `optional`, defaults to 1,000):
            Number of dimensions to use for the sequences latent code.
        set_seq_size (:obj:`int`, `optional`, defaults to 60):
            NOTE: Here it is the set sequence size, every sample must be padded to be equal to this length.
        additional_latent_models (:obj:`list[nn.Module]`, `optional`, defaults to empty list):
            List of models that take the latent code and return a loss.
            Use this to condition the latent code on another model optimising the latent space further.
        t5_config_kwargs
            These are sent to `T5Config` to configure the T5 Model.
    """
    model_type = "t5_vae"
    is_composition = True

    def __init__(self, latent_size=1_000, set_seq_size=60, additional_latent_models=[], **t5_config_kwargs):
        t5_config_kwargs["n_positions"] = set_seq_size
        super().__init__(**t5_config_kwargs)
        self.t5_config = T5Config(**t5_config_kwargs)
        self.latent_size = latent_size
        self.set_seq_size = set_seq_size
        self.additional_latent_models = additional_latent_models

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default `to_dict()` from `PretrainedConfig`.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["t5_config"] = self.t5_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
