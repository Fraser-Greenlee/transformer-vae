from typing import Optional
import numpy as np
import datasets


class SimpleMetrics(datasets.Metric):
    """
    Metrics that don't use the writer.
    """

    def compute(self, *args, **kwargs) -> Optional[dict]:
        """Compute the metrics.

        Args:
            We disallow the usage of positional arguments to prevent mistakes
            `predictions` (Optional list/array/tensor): predictions
            `references` (Optional list/array/tensor): references
            `**kwargs` (Optional other kwargs): will be forwared to the metrics :func:`_compute` method (see details in the docstring)

        Return:
            Dictionnary with the metrics if this metric is run on the main process (process_id == 0)
            None if the metric is not run on the main process (process_id != 0)
        """
        if args:
            raise ValueError("Please call `compute` using keyword arguments.")

        predictions = kwargs.pop("predictions", None)
        references = kwargs.pop("references", None)

        assert self.process_id == 0, "Not implimented distributed metrics yet."

        with datasets.utils.temp_seed(self.seed):
            output = self._compute(predictions=predictions, references=references, **kwargs)

        return output


_DESCRIPTION = """
What percentage of greedily decoded sequences would be completely correct?
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: Label logits, as returned by a model.
    references: Ground truth labels.
Returns:
    accuracy: Accuracy score.
"""


class SequenceAccuracy(SimpleMetrics):
    name = "sequence-accuracy"

    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("float32"),
                    "references": datasets.Value("int32"),
                }
            ),
            citation=None,
        )

    def _compute(self, predictions, references):
        predicted_tokens = np.argmax(predictions, axis=2)
        correct_sequences = (predicted_tokens != references).sum(axis=1) == 0
        return {
            self.name: correct_sequences.sum() / references.shape[0],
        }


_DESCRIPTION = """
What percentage of tokens would be correct?
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: Label logits, as returned by a model.
    references: Ground truth labels. (-100 should be padding tokens)
Returns:
    accuracy: Accuracy score.
"""


class TokenAccuracy(SimpleMetrics):
    name = "token-accuracy"

    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("float32"),
                    "references": datasets.Value("int32"),
                }
            ),
            citation=None,
        )

    def _compute(self, predictions, references):
        predicted_tokens = np.argmax(predictions, axis=2)
        return {
            self.name: (predicted_tokens == references).sum() / (references.size - (references == -100).sum()),
        }


METRICS_MAP = {
    SequenceAccuracy.name: SequenceAccuracy,
    TokenAccuracy.name: TokenAccuracy,
}
