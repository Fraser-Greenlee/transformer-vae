from transformers.integrations import WandbCallback
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
            raise ValueError("Need to be sent `model` to update global step.")
        model.global_step = state.global_step


class WandbCallbackUseModelLogs(WandbCallback):
    """
    Adds model's internal logs to allow logging extra losses.
    """

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs:
            logs = {**logs, **model.get_latest_logs()}
        super().on_log(args, state, control, model=None, logs=None, **kwargs)
