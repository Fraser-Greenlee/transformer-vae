import torch
import logging
from typing import Optional, Dict
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.dataloader import DataLoader

from datasets.features import ClassLabel
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
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        mini_eval_dataloader_iter = iter(
            DataLoader(
                eval_dataset,
                sampler=RandomSampler(eval_dataset),
                batch_size=2,
                collate_fn=self.data_collator,
            )
        )

        samples = self._prepare_inputs(next(mini_eval_dataloader_iter))
        latents = self.model(**samples).latnet
        start_latent, end_latent = latents[0].view(1, -1), latents[1].view(1, -1)
        latent_diff = end_latent - start_latent

        table = wandb.Table(columns=["Interpolation Ratio", "Text"])
        for i in range(11):
            ratio = i / 10
            latent_point = start_latent + ratio * latent_diff
            # need to give bos_token_id even for encoder_decoder models like T5
            # TODO this may not work for Funnel-VAE
            generation = self.model.generate(latent=latent_point, bos_token_id=0)
            import pdb; pdb.set_trace()
            table.add_data(ratio, generation)
        wandb.log({"interpolate points": table})

    def _random_samples(self):
        table = wandb.Table(columns=["Text"])
        latent_points = torch.randn(10, self.model.config.latent_size)
        for i in range(latent_points.size(0)):
            import pdb; pdb.set_trace()
            table.add_data(
                self.model.generate(latent=latent_points[i], bos_token_id=0)
            )
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
        if is_wandb_available():
            with torch.no_grad():
                self.model.eval()
                self._interpolate_samples(eval_dataset)
                self._random_samples()
                # TODO add t-SNE clustering with class labels

        output_metrics = super().evaluate(eval_dataset=eval_dataset)
        return output_metrics
