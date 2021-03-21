"""Wikipedia Sentences"""

from __future__ import absolute_import, division, print_function

import json

import datasets


_DESCRIPTION = """\
Dataset of sentences from Wikipedia (from the [Optimus paper](https://arxiv.org/abs/2004.04092)).
Each is of mex 64 words & <=256 GPT2 tokens.
Each row is a tokenised sentence.
{'token_ids': '{gpt2 token ids}'}
This is to test the semantics of a Transformer-VAEs latent space by interpolating on sentences.
"""

_MAX_SEGMENTS = 4
_TRAIN_DOWNLOAD_URL = r"https://storage.googleapis.com/t-vae/wikipedia_json_64_filtered_segment_{0}.zip"


class PythonLines(datasets.GeneratorBasedBuilder):
    """Python lines dataset."""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    'token_ids': datasets.Value("list of integers"),
                }
            ),
            homepage="https://github.com/Fraser-Greenlee/transformer-vae",
        )

    def _split_generators(self, dl_manager):
        import pdb
        pdb.set_trace()
        assert(self.config.segment_num <= _MAX_SEGMENTS), f'Segment does not exist, requested segment {self.config.segment_num}, but max segment num is {_MAX_SEGMENTS}'
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL.format(self.config.segment_num))
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}),
        ]

    def _generate_examples(self, filepath):
        """Generate examples."""
        with open(filepath, encoding="utf-8") as json_lines_file:
            data = []
            for line in json_lines_file:
                data.append(json.loads(line))

            for id_, row in enumerate(data):
                yield id_, row
