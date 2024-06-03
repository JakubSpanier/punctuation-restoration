import os
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
import random
from pathlib import Path
from typing import Tuple, Final

BASE_DIR: Final[Path] = Path(os.getcwd())

SEED: Final[int] = 1353
TRAIN_TO_TEST_DEV_RATIO: Final[float] = 0.8
DEV_TO_TEST_RATIO: Final[float] = 0.5


@dataclass
class TsvFileData:
    words: list[list[str]] = field(default_factory=lambda: [[]])
    tags: list[list[str]] = field(default_factory=lambda: [[]])
    spaces_after: list[list[str]] = field(default_factory=lambda: [[]])

    def __iter__(self):
        return iter(zip(self.words, self.tags, self.spaces_after))

    def append(self, word: str, tag: str, space: str) -> None:
        self.words[-1].append(word)
        self.tags[-1].append(tag)
        self.spaces_after[-1].append(space)

    def append_next_lists(self) -> None:
        self.words.append([])
        self.tags.append([])
        self.spaces_after.append([])

    def to_data_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "words": word,
                "labels": tag,
                "times": space_after,
                "sentence_id": id_,
            }
            for id_, (words, tags, spaces_after) in enumerate(self)
            for word, tag, space_after in zip(words, tags, spaces_after)
        )


class DataFrameSplitter:
    def __init__(self, data_file: Path, out_file: Path, split_to_files: bool = False):
        """
        Initializes the DataFrameSplitter with the given parameters.

        :param data_file: Path to the input data file.
        :param out_file: Path to the output file or directory.
        :param split_to_files: Flag to split data into multiple files.
        """
        self.data_file = data_file
        self.out_file = out_file
        self.split_to_files = split_to_files
        self.seed = SEED
        self.data = TsvFileData()
        random.seed(SEED)
        np.random.seed(SEED)

    def run(self):
        """
        Runs the data processing and splitting based on initialized parameters.
        """
        df = self._process_file(self.data_file)

        if self.split_to_files:
            train_data, dev_test_data = self._split_dataframe(
                df, train_prop=TRAIN_TO_TEST_DEV_RATIO
            )
            dev_data, test_data = self._split_dataframe(
                dev_test_data, train_prop=DEV_TO_TEST_RATIO
            )

            self._save_to_file(train_data, suffix=f"_train_{self.seed}")
            self._save_to_file(dev_data, suffix=f"_dev_{self.seed}")
            self._save_to_file(test_data, suffix=f"_test_{self.seed}")
        else:
            self._save_to_file(df)

    def _process_file(self, path: Path) -> pd.DataFrame:
        """
        Processes the input file and converts it into a DataFrame.

        :param path: The path to the input data file.
        :return: The processed DataFrame.
        """

        with open(path) as f:
            for line in f:
                self._read_line(line=line)
            if not self.data.words[-1]:
                self.data.words = self.data.words[:-1]
                self.data.tags = self.data.tags[:-1]
                self.data.spaces_after = self.data.spaces_after[:-1]

        return self.data.to_data_frame()

    def _read_line(self, line: str) -> None:
        if line.startswith("-DOCSTART-"):
            return

        stripped_line = line.strip()
        if not stripped_line:
            self.data.append_next_lists()
            return

        try:
            word, tag, space = stripped_line.split("\t")
        except ValueError:
            word, tag = stripped_line.split("\t")
            space = " "
        self.data.append(word=word, tag=tag, space=space)

    @staticmethod
    def _split_dataframe(
        data: pd.DataFrame, train_prop: float = 0.9
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits a DataFrame into training and testing sets based on sentence_id.

        :param data: The DataFrame to split.
        :param train_prop: The proportion of the data to use for training.
        :return: A tuple containing the training and testing DataFrames.
        """
        sentence_ids = data.sentence_id.unique().tolist()
        train_count = round(len(sentence_ids) * train_prop)
        train_ids = random.sample(sentence_ids, train_count)

        train_data = data.loc[data.sentence_id.isin(train_ids)]
        test_data = data.loc[~data.sentence_id.isin(train_ids)]

        return train_data, test_data

    def _save_to_file(self, data: pd.DataFrame, suffix: str = "") -> None:
        """
        Saves the DataFrame to a file.

        :param data: The DataFrame to save.
        :param out_file: The base name of the output file.
        :param suffix: The suffix to add to the file name.
        """

        file_path = self.out_file.parent / f"{self.out_file.stem}{suffix}.tsv"
        data.to_csv(file_path, sep="\t", index=False)


if __name__ == "__main__":
    files = (
        (
            BASE_DIR / Path("parsed_data/original_train.conll"),
            BASE_DIR / Path("parsed_data/original_train.tsv"),
        ),
        (
            BASE_DIR / Path("parsed_data/original_test-A.conll"),
            BASE_DIR / Path("parsed_data/original_test-A.tsv"),
        ),
        (
            BASE_DIR / Path("parsed_data/text.rest/rest.conll"),
            BASE_DIR / Path("parsed_data/text.rest/rest.tsv"),
        ),
    )
    for data_file, out_file in files:
        splitter = DataFrameSplitter(data_file=data_file, out_file=out_file)
        splitter.run()
