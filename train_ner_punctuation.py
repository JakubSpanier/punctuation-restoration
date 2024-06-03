import os
from logging import getLogger
from pathlib import Path

import pandas as pd
import torch
import time
from typing import Final
from simpletransformers.ner import NERArgs, NERModel

from enums import Label
from metric_functions import METRICS

logger = getLogger(__name__)

WANDB_PROJECT: Final[str] = "punctuation-restoration"
WEIGHTS = None
EPOCHS: Final[int] = 2
LEARNING_RATE: Final[float] = 2e-5
BATCH_SIZE: Final[int] = 12
ACC: Final[int] = 1
MODEL_NAME: Final[str] = "allegro/herbert-base-cased"
MODEL_TYPE: Final[str] = "herbert"
WARMUP_STEPS: Final[int] = 0
EVAL_STEPS: Final[int] = 200
EVAL_DURING_TRAINING: Final[bool] = True
MAX_SEQ_LEN: Final[int] = 256
USE_DICE: Final[bool] = False
USE_FOCAL: Final[bool] = False
FOCAL_ALPHA: Final[float] = 0.25
SEED: Final[int] = 2
EARLY_STOPPING_METRIC: Final[str] = "f1_weighted"


class NERTrainer:
    def __init__(self, train_path: Path, eval_path: Path):
        self.train_path = train_path
        self.eval_path = eval_path

        self.labels = Label.list()
        self.ner_args = self._setup_ner_args()

    @classmethod
    def _setup_ner_args(cls) -> NERArgs:
        """setting arguments for NER model"""
        ner_args = NERArgs()
        ner_args.early_stopping_metric = EARLY_STOPPING_METRIC
        ner_args.early_stopping_metric_minimize = False
        ner_args.model_type = MODEL_TYPE
        ner_args.model_name = MODEL_NAME
        # ner_args.wandb_project = WANDB_PROJECT
        # ner_args.wandb_kwargs = {"settings": wandb.Settings(start_method="thread")}
        ner_args.train_batch_size = BATCH_SIZE
        ner_args.eval_batch_size = BATCH_SIZE
        ner_args.gradient_accumulation_steps = ACC
        ner_args.learning_rate = LEARNING_RATE
        ner_args.num_train_epochs = EPOCHS
        ner_args.evaluate_during_training = EVAL_DURING_TRAINING
        ner_args.evaluate_during_training_steps = EVAL_STEPS
        ner_args.max_seq_length = MAX_SEQ_LEN
        ner_args.manual_seed = SEED
        ner_args.warmup_steps = WARMUP_STEPS
        ner_args.save_eval_checkpoints = False
        ner_args.use_multiprocessing = False
        ner_args.use_multiprocessing_for_evaluation = False

        if USE_DICE:
            ner_args.loss_type = "dice"
            ner_args.loss_args = {
                "smooth": 0.001,
                "square_denominator": True,
                "with_logits": True,
                "ohem_ratio": 0.0,
                "alpha": 0,
                "reduction": "mean",
                "index_label_position": True,
            }
        if USE_FOCAL:
            ner_args.loss_type = "focal"
            ner_args.loss_args = {
                "alpha": FOCAL_ALPHA,
                "gamma": 2,
                "reduction": "mean",
                "eps": 1e-6,
                "ignore_index": -100,
            }

        return ner_args

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_data = pd.read_csv(self.train_path, sep="\t", header=0)
        eval_data = pd.read_csv(self.eval_path, sep="\t", header=0)

        sentences_to_delete = train_data[
            ~train_data.labels.isin(self.labels)
        ].sentence_id
        train_data = train_data.loc[~train_data.sentence_id.isin(sentences_to_delete)]

        sentences_to_delete = eval_data[~eval_data.labels.isin(self.labels)].sentence_id
        eval_data = eval_data.loc[~eval_data.sentence_id.isin(sentences_to_delete)]

        return train_data, eval_data

    def train(self):
        train_data, eval_data = self.load_data()

        train_data.words = train_data.words.astype(str)
        eval_data.words = eval_data.words.astype(str)

        logger.info("Labels:", self.labels)
        start = time.time()
        output_dir = Path(f"models/{MODEL_NAME}_{start}")

        self.ner_args.output_dir = output_dir
        self.ner_args.best_model_dir = output_dir / Path("best_model")
        model = NERModel(
            MODEL_TYPE,
            MODEL_NAME,
            labels=self.labels,
            args=self.ner_args,
            weight=WEIGHTS,
            use_cuda=torch.cuda.is_available(),
        )

        model.train_model(
            train_data,
            output_dir=output_dir,
            show_running_loss=True,
            eval_data=eval_data,
            **METRICS,
        )


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    train_path = Path("parsed_data/text.rest/rest_splitted.tsv")
    eval_path = Path("parsed_data/original_train_splitted.tsv")
    trainer = NERTrainer(train_path=train_path, eval_path=eval_path)
    trainer.train()
