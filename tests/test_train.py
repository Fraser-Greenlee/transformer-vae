import logging
import sys
from unittest.mock import patch
import torch
from transformers.testing_utils import TestCasePlus, torch_device

from transformer_vae.train import main


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()


class TrainTests(TestCasePlus):
    def test_train_txt(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            train.py
            --train_file ./tests/fixtures/line_by_line_max_len_3.txt
            --validation_file ./tests/fixtures/line_by_line_max_len_3.txt
            --do_train
            --do_eval
            --per_device_train_batch_size 4
            --per_device_eval_batch_size 4
            --num_train_epochs 2
            --set_seq_size 4
            --latent_size 2
            --transformer_name t5-small
            --output_dir {tmp_dir}
            --overwrite_output_dir
            """.split()

        if torch.cuda.device_count() > 1:
            # Skipping because there are not enough batches to train the model + would need a drop_last to work.
            return

        if torch_device != "cuda":
            testargs.append("--no_cuda")

        with patch.object(sys, "argv", testargs):
            result = main()
            self.assertAlmostEqual(result["epoch"], 2.0)

    def test_train_csv(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            train.py
            --train_file ./tests/fixtures/multiline_max_len_4.csv
            --validation_file ./tests/fixtures/multiline_max_len_4.csv
            --do_train
            --do_eval
            --per_device_train_batch_size 5
            --per_device_eval_batch_size 5
            --num_train_epochs 2
            --set_seq_size 5
            --latent_size 2
            --transformer_name t5-small
            --output_dir {tmp_dir}
            --overwrite_output_dir
            """.split()

        if torch.cuda.device_count() > 1:
            # Skipping because there are not enough batches to train the model + would need a drop_last to work.
            return

        if torch_device != "cuda":
            testargs.append("--no_cuda")

        with patch.object(sys, "argv", testargs):
            result = main()
            self.assertAlmostEqual(result["epoch"], 2.0)

    def test_train_json(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            train.py
            --train_file ./tests/fixtures/max_len_3.json
            --validation_file ./tests/fixtures/max_len_3.json
            --do_train
            --do_eval
            --per_device_train_batch_size 5
            --per_device_eval_batch_size 5
            --num_train_epochs 2
            --set_seq_size 4
            --latent_size 2
            --transformer_name t5-small
            --output_dir {tmp_dir}
            --overwrite_output_dir
            """.split()

        if torch.cuda.device_count() > 1:
            # Skipping because there are not enough batches to train the model + would need a drop_last to work.
            return

        if torch_device != "cuda":
            testargs.append("--no_cuda")

        with patch.object(sys, "argv", testargs):
            result = main()
            self.assertAlmostEqual(result["epoch"], 2.0)

    def test_train_funnel(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            train.py
            --train_file ./tests/fixtures/line_by_line_max_len_3.txt
            --validation_file ./tests/fixtures/line_by_line_max_len_3.txt
            --do_train
            --do_eval
            --per_device_train_batch_size 4
            --per_device_eval_batch_size 4
            --num_train_epochs 2
            --set_seq_size 8
            --encoded_seq_size 2
            --latent_size 2
            --transformer_type funnel
            --transformer_name funnel-transformer/small
            --output_dir {tmp_dir}
            --overwrite_output_dir
            """.split()

        if torch.cuda.device_count() > 1:
            # Skipping because there are not enough batches to train the model + would need a drop_last to work.
            return

        if torch_device != "cuda":
            testargs.append("--no_cuda")

        with patch.object(sys, "argv", testargs):
            result = main()
            self.assertAlmostEqual(result["epoch"], 2.0)

    def test_train_funnel_t5(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            train.py
            --train_file ./tests/fixtures/line_by_line_max_len_3.txt
            --validation_file ./tests/fixtures/line_by_line_max_len_3.txt
            --do_train
            --do_eval
            --per_device_train_batch_size 4
            --per_device_eval_batch_size 4
            --num_train_epochs 2
            --set_seq_size 8
            --encoder_model full-n-tokens
            --decoder_model full-tokens
            --n_latent_tokens 8
            --encoded_seq_size 2
            --latent_size 2
            --transformer_type funnel-t5
            --transformer_name funnel-transformer/intermediate
            --transformer_decoder_name t5-base
            --output_dir {tmp_dir}
            --overwrite_output_dir
            """.split()

        if torch.cuda.device_count() > 1:
            # Skipping because there are not enough batches to train the model + would need a drop_last to work.
            return

        if torch_device != "cuda":
            testargs.append("--no_cuda")

        with patch.object(sys, "argv", testargs):
            result = main()
            self.assertAlmostEqual(result["epoch"], 2.0)

    def test_train_1st_token(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            train.py
            --train_file ./tests/fixtures/line_by_line_max_len_3.txt
            --validation_file ./tests/fixtures/line_by_line_max_len_3.txt
            --do_train
            --do_eval
            --per_device_train_batch_size 4
            --per_device_eval_batch_size 4
            --num_train_epochs 2
            --set_seq_size 8
            --encoded_seq_size 2
            --latent_size 2
            --transformer_type t5
            --transformer_name t5-small
            --encoder_model 1st-token
            --output_dir {tmp_dir}
            --overwrite_output_dir
            """.split()

        if torch.cuda.device_count() > 1:
            # Skipping because there are not enough batches to train the model + would need a drop_last to work.
            return

        if torch_device != "cuda":
            testargs.append("--no_cuda")

        with patch.object(sys, "argv", testargs):
            result = main()
            self.assertAlmostEqual(result["epoch"], 2.0)

    def test_train_mini_mmd_batch_size(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            train.py
            --train_file ./tests/fixtures/line_by_line_max_len_3.txt
            --validation_file ./tests/fixtures/line_by_line_max_len_3.txt
            --do_train
            --do_eval
            --per_device_train_batch_size 4
            --per_device_eval_batch_size 4
            --mmd_batch_size 2
            --num_train_epochs 2
            --set_seq_size 4
            --latent_size 2
            --transformer_type t5
            --transformer_name t5-small
            --output_dir {tmp_dir}
            --overwrite_output_dir
            """.split()

        if torch.cuda.device_count() > 1:
            # Skipping because there are not enough batches to train the model + would need a drop_last to work.
            return

        if torch_device != "cuda":
            testargs.append("--no_cuda")

        with patch.object(sys, "argv", testargs):
            result = main()
            self.assertAlmostEqual(result["epoch"], 2.0)

    def test_train_python_syntax_seq_check(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            train.py
            --train_file ./tests/fixtures/line_by_line_max_len_3.txt
            --validation_file ./tests/fixtures/line_by_line_max_len_3.txt
            --do_train
            --do_eval
            --sample_from_latent
            --per_device_train_batch_size 4
            --per_device_eval_batch_size 4
            --num_train_epochs 1
            --set_seq_size 4
            --latent_size 2
            --transformer_name t5-small
            --output_dir {tmp_dir}
            --overwrite_output_dir
            --seq_check python
            """.split()

        if torch.cuda.device_count() > 1:
            # Skipping because there are not enough batches to train the model + would need a drop_last to work.
            return

        if torch_device != "cuda":
            testargs.append("--no_cuda")

        with patch.object(sys, "argv", testargs):
            result = main()
            self.assertAlmostEqual(result["epoch"], 1.0)

    def test_train_non_vae(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            train.py
            --train_file ./tests/fixtures/line_by_line_max_len_3.txt
            --validation_file ./tests/fixtures/line_by_line_max_len_3.txt
            --do_train
            --do_eval
            --per_device_train_batch_size 4
            --per_device_eval_batch_size 4
            --num_train_epochs 2
            --set_seq_size 4
            --dont_use_reg_loss
            --encoder_model full-1st-token
            --decoder_model full-tokens
            --transformer_name t5-small
            --output_dir {tmp_dir}
            --overwrite_output_dir
            """.split()

        if torch.cuda.device_count() > 1:
            # Skipping because there are not enough batches to train the model + would need a drop_last to work.
            return

        if torch_device != "cuda":
            testargs.append("--no_cuda")

        with patch.object(sys, "argv", testargs):
            result = main()
            self.assertAlmostEqual(result["epoch"], 2.0)

    def test_train_unsupervised_classification(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            train.py
            --dataset_name=Fraser/news-category-dataset
            --text_column=headline
            --classification_column=category_num
            --do_eval
            --per_device_train_batch_size 2
            --per_device_eval_batch_size 2
            --max_validation_size 100
            --eval_steps 4
            --encoder_model full-1st-token
            --decoder_model full-tokens
            --latent_size 2
            --transformer_name t5-small
            --output_dir {tmp_dir}
            --overwrite_output_dir
            """.split()

        if torch.cuda.device_count() > 1:
            # Skipping because there are not enough batches to train the model + would need a drop_last to work.
            return

        if torch_device != "cuda":
            testargs.append("--no_cuda")

        with patch.object(sys, "argv", testargs):
            result = main()
            self.assertGreater(result["eval_loss"], 0.0)
            self.assertNotIn("epoch", result)

    def test_train_n_tokens_model(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            train.py
            --train_file ./tests/fixtures/line_by_line_max_len_3.txt
            --do_train
            --per_device_train_batch_size 2
            --num_train_epochs 1
            --set_seq_size 4
            --encoder_model full-n-tokens
            --n_latent_tokens 2
            --decoder_model full-tokens
            --latent_size 2
            --transformer_name t5-small
            --output_dir {tmp_dir}
            --overwrite_output_dir
            """.split()

        if torch.cuda.device_count() > 1:
            # Skipping because there are not enough batches to train the model + would need a drop_last to work.
            return

        if torch_device != "cuda":
            testargs.append("--no_cuda")

        with patch.object(sys, "argv", testargs):
            main()

    def test_train_unsupervised_classification_agnews(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            train.py
            --dataset_name=ag_news
            --classification_column=label
            --do_train
            --max_steps=10
            --validation_name=test
            --test_classification
            --per_device_train_batch_size 2
            --per_device_eval_batch_size 2
            --max_validation_size 100
            --encoder_model full-1st-token
            --decoder_model full-tokens
            --latent_size 2
            --transformer_name t5-small
            --output_dir {tmp_dir}
            --overwrite_output_dir
            """.split()

        if torch.cuda.device_count() > 1:
            # Skipping because there are not enough batches to train the model + would need a drop_last to work.
            return

        if torch_device != "cuda":
            testargs.append("--no_cuda")

        with patch.object(sys, "argv", testargs):
            main()
