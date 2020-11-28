import pdb
import torch
import logging
from typing import Optional, Dict
from torch.utils.data.dataset import Dataset

from dataset.features import ClassLabel
from transformers import trainer as trainer_script
from transformers.integrations import WandbCallback, is_wandb_available
from transformer_vae.trainer_callback import WandbCallbackUseModelLogs


logger = logging.getLogger(__name__)


if WandbCallback in trainer_script.DEFAULT_CALLBACKS:
    # Allow tracking extra training losses via the model's `get_latest_logs` method
    trainer_script.DEFAULT_CALLBACKS.remove(WandbCallback)
    trainer_script.DEFAULT_CALLBACKS.append(WandbCallbackUseModelLogs)
    import wandb
else:
    logger.warn("Not using Weights and Biasis, this will give you incomplete logs.")


class VAE_Trainer(trainer_script.Trainer):
    def _interpolate_samples(self, eval_dataset):
        pdb.set_trace()
        table = wandb.Table(columns=["Interpolation Ratio", "Text"])

        start_sample_text, end_sample_text = [eval_dataset.sample()["text"] for _ in range(2)]
        start_latent, end_latent = (
            self.model(input_ids=start_sample_text).latent_code,
            self.model(input_ids=end_sample_text).latent_code,
        )
        latent_diff = end_latent - start_latent

        for i in range(11):
            ratio = i / 10
            latent_point = start_latent + ratio * latent_diff
            table.add_data(ratio, self.model.generate(latent_code=latent_point))
        wandb.log({"interpolate points": table})

    def _random_samples(self):
        pdb.set_trace()
        table = wandb.Table(columns=["Text"])
        latent_points = torch.randn(10, self.model.config.latent_size)
        for i in range(latent_points.size(0)):
            table.add_data(self.model.generate(latent_code=latent_points[i]))
        wandb.log({"random points": table})

    def _clustering(self, eval_dataset, class_column_name):
        if class_column_name is None:
            for key, val in eval_dataset.features['category_num'].items():
                if type(val) is ClassLabel:
                    class_column_name = key
        # TODO plot t-SNE with points coloured for each category

    def evaluate(self, eval_dataset: Optional[Dataset] = None, class_column_name=None) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        Adds extra VAE tests:
        - Interpolation between samples in latent space.
        - Random latent codes from normal distribution.
        if class column provided?
        - tSNE plots with class-label colouring.
        """
        output_metrics = super().evaluate(eval_dataset=eval_dataset)
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        if eval_dataset is None:
            raise ValueError('No eval dataset available.')
        if eval_dataset and is_wandb_available():
            self._interpolate_samples(eval_dataset)
            self._random_samples()
            # TODO add unsupervised classification
        return output_metrics
