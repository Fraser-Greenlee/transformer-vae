from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerControl,
    TrainerState,
)


class TellModelGlobalStep(TrainerCallback):
    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        optimizer=None,
        lr_scheduler=None,
        train_dataloader=None,
        eval_dataloader=None,
        **kwargs,
    ):
        if not model:
            raise ValueError('Need to be sent model to update global step.')
        model.global_step = state.global_step
