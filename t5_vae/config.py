from transformers.configuration_utils import PretrainedConfig


class T5_VAE_Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of :class:`~t5_vae.T5_VAE`.
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
        additional_latent_models (:obj:`str`, `optional`, defaults to empty list):
            List of models that take the latent code and return a loss.
            Use this to condition the latent code on another model optimising the latent space further.
    """
    model_type = "t5_vae"

    def __init__(self, t5_model_name="t5-base", latent_size=1_000, set_seq_size=60, additional_latent_models=[]):
        self.t5_model_name = t5_model_name
        self.latent_size = latent_size
        self.set_seq_size = set_seq_size
        self.additional_latent_models = additional_latent_models
