'''
    I wanted to use a critic model to improve interpolation performance.

    Sadly none of these methods worked.
'''
import torch
from torch import nn
from transformers import AutoModelForSeq2SeqLM, AutoModelForMaskedLM


class Critic(nn.Module):
    """
    Model uses an transformer encoder to judge if decoder hidden states are from real VS interpolated latent points.
    """
    def __init__(self, config):
        super().__init__()
        try:
            self.critic = AutoModelForSeq2SeqLM.from_config(config).encoder
        except ValueError:
            # handle funnel critic model
            self.critic = AutoModelForMaskedLM.from_config(config).funnel.encoder
        self.fc = nn.Linear(config.d_model, 1)
        self.activation = nn.Sigmoid()
        self.loss = nn.MSELoss()

    def forward(self, hidden_state, targets=None):
        attention_mask = torch.ones(hidden_state.size()[:-1], device=hidden_state.device)
        final_hidden = self.critic(hidden_state, attention_mask=attention_mask).last_hidden_state
        score = 0.5 * self.activation(self.fc(final_hidden[:, 0]))
        if targets is not None:
            return self.loss(score, targets)
        return score


class CriticMean(Critic):
    """
    Takes the mean of each tokens score instead of encouraging a seq level score.
    """
    def forward(self, hidden_state, targets=None):
        attention_mask = torch.ones(hidden_state.size()[:-1], device=hidden_state.device)
        final_hidden = self.critic(hidden_state, attention_mask=attention_mask).last_hidden_state
        score = 0.5 * self.activation(self.fc(final_hidden)).mean(dim=1)
        if targets is not None:
            return self.loss(score, targets)
        return score


class CriticMeanNoAct(Critic):
    """
    Takes the mean of each tokens score & doesn't use an activation function.
    """
    def forward(self, hidden_state, targets=None):
        attention_mask = torch.ones(hidden_state.size()[:-1], device=hidden_state.device)
        final_hidden = self.critic(hidden_state, attention_mask=attention_mask).last_hidden_state
        score = self.fc(final_hidden).mean(dim=1)
        if targets is not None:
            return self.loss(score, targets)
        return score


CRITIC = {
    '': Critic,
    'mean': CriticMean,
    'mean_no_act': CriticMeanNoAct,
}
