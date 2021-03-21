"""Wikipedia Sentences"""

from __future__ import absolute_import, division, print_function

import os
import json

import datasets


_DESCRIPTION = """\
Dataset of sentences from Wikipedia (from the [Optimus paper](https://arxiv.org/abs/2004.04092)).
Each is of mex 64 words & <=256 GPT2 tokens.
Each row is a tokenised sentence.
{'token_ids': '{gpt2 token ids}'}
This is to test the semantics of a Transformer-VAEs latent space by interpolating on sentences.
"""

NUM_SEGMENTS = 5
_TRAIN_DOWNLOAD_URL = r"https://storage.googleapis.com/t-vae/wikipedia_json_64_filtered_segment_{0}.zip"


class WikiSentencesConfig(datasets.BuilderConfig):
    """BuilderConfig for WikiSentences."""

    def __init__(self, segment=None, **kwargs):
        """BuilderConfig for WikiSentences.
        Args:
          segment_num: keyword argument to specify the segment of the dataset to load
          **kwargs: keyword arguments forwarded to super.
        """
        self.segment = segment
        super(WikiSentencesConfig, self).__init__(**kwargs)


class WikiSentences(datasets.GeneratorBasedBuilder):
    """Sentences from Wikipedia."""

    BUILDER_CONFIGS = [
        WikiSentencesConfig(
            name=f"segment_{i}",
            description=f"Segment {i+1}/{NUM_SEGMENTS} of WikiSentences Dataset for interpolating on natural language.",
            segment=i
        ) for i in range(NUM_SEGMENTS)
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    'token_ids': [datasets.Value("int32")],
                }
            ),
            homepage="https://github.com/Fraser-Greenlee/transformer-vae",
        )

    def _split_generators(self, dl_manager):
        assert(self.config.segment < NUM_SEGMENTS), f'Segment does not exist, requested segment {self.config.segment}, but max segment num is {NUM_SEGMENTS - 1}'
        folder_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL.format(self.config.segment))
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": os.path.join(folder_path, 'segment_output.jsonl')}
            ),
        ]

    def _generate_examples(self, filepath):
        """Generate examples."""
        with open(filepath, encoding="utf-8") as json_lines_file:
            for id_, line in enumerate(json_lines_file):
                yield id_, json.loads(line)
