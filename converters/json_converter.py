import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Self


@dataclass
class WordData:
    punctuation: str
    space_after: bool
    word: str

    @classmethod
    def from_dict(cls, data: dict[str, str | bool]) -> Self:
        return cls(
            punctuation=data["punctuation"],
            space_after=data["space_after"],
            word=data["word"],
        )


@dataclass
class TextData:
    title: str
    words: list[WordData]

    @classmethod
    def from_dict(cls, data: dict[str, str | list[dict[str, str | bool]]]) -> Self:
        return cls(
            title=data["title"],
            words=[WordData.from_dict(data=word) for word in data["words"]],
        )


class JsonToTsvConverter:
    def __init__(
        self,
        train_path: Optional[Path] = None,
        test_path: Optional[Path] = None,
        data_paths: Optional[list[Path]] = None,
        save_path: Optional[Path] = None,
    ) -> None:
        """

        :param train_path: Path to train in.tsv
        :param test_path: Path to test in.tsv
        :param data_paths: Paths to dirs with JSONs
        :param save_path: Path to directory
        """
        # a or b is just more pythonic way to do a if a else b
        self.train_path = train_path or Path("data/train/in.tsv")
        self.test_path = test_path or Path("data/test-A/in.tsv")
        default_data_paths = [
            Path("data/text.rest/wikinews/all/json"),
            Path("data/text.rest/wikitalks/all/json"),
        ]
        self.data_paths = data_paths or default_data_paths
        self.save_path = save_path or Path("data/text.rest")

    def convert(self) -> None:
        """
        convert .json files into .tsv files, alike transcripts
        """
        train_paths, test_paths, rest_paths = self._get_json_paths()

        for name, paths in [
            ("train", train_paths),
            ("test", test_paths),
            ("rest", rest_paths),
        ]:
            self._write_output(name, paths)

    def _get_json_paths(self) -> tuple[list[Path], list[Path], list[Path]]:
        """
        get all the wikipunct text ids from json files
        put them in the train, test list if they are present in train or test transcripts
        sort them in a strange way.
        others are put in the rest list and shuffle
        :return:

        Returns:
            ["wikinews228460", "wikitews112112"], ["wikitalks23456", "wikitalks33456"], []
        """
        train_names = self._parse_wikipunct_text_ids(self.train_path)
        test_names = self._parse_wikipunct_text_ids(self.test_path)

        train_paths, test_paths, rest_paths = [], [], []
        for path in self.data_paths:
            for json_path in path.glob("*.json"):
                wikipunct_text_id = json_path.stem
                if wikipunct_text_id in train_names:
                    train_paths.append(json_path)
                elif wikipunct_text_id in test_names:
                    test_paths.append(json_path)
                else:
                    rest_paths.append(json_path)

        train_paths = self._sort_paths(train_paths, train_names)
        test_paths = self._sort_paths(test_paths, test_names)

        random.seed(0)
        random.shuffle(rest_paths)

        return train_paths, test_paths, rest_paths

    @staticmethod
    def _parse_wikipunct_text_ids(path: Path) -> list[str]:
        """
        parse wikipunct text ids from the .tsv file transcripts
        :param path:
        :return:

        Returns:
            ["wikinews228460", "wikitalks23456"]
        """
        names = []
        with open(path) as file:
            for line in file:
                line = line.strip()
                name, _ = line.split("\t")
                names.append(name)
        return names

    @staticmethod
    def _sort_paths(paths: list[Path], wikipunct_text_ids: list[str]) -> list[Path]:
        """
        sort by the order in which, wikipunct_text_ids are present in .tsv files

        :param paths:
        :param wikipunct_text_ids:
        :return:
        """
        sorted_paths = []
        for wikipunct_text_id in wikipunct_text_ids:
            for path in paths:
                if wikipunct_text_id == path.stem:
                    sorted_paths.append(path)
                    break
        return sorted_paths

    def _write_output(self, name: str, paths: list[Path]) -> None:
        expected_path = Path(self.save_path / f"{name}_expected.tsv")
        in_path = Path(self.save_path / f"{name}_in.tsv")
        with open(expected_path, "w") as out_expected, open(in_path, "w") as out_in:
            for path in paths:
                json_in, json_expected = self._load_json(path)
                out_in.write(f"{path.stem}\t{json_in}\n")
                out_expected.write(f"{json_expected}\n")

    def _load_json(self, path: Path) -> tuple[str, str]:
        """
        Loads a .json file and processes its content to generate two cleaned text strings

        :param path:
        :return:
        """
        text_in, text_exp, text_intact = self._build_texts(path=path)
        text_in = self._normalize_text_in(text=text_in)
        text_exp = self._normalize_text_exp(text=text_exp)

        return text_in, text_exp

    @staticmethod
    def _build_texts(path: Path) -> tuple[str, str, str]:
        """
        build text without punctuation, with, and without any manipulation

        :param path:
        :return:

        Returns:
            "Proszę Pana Jana", "Proszę, Pana Jana", "Proszę, Pana ^ Jana"
        """

        with open(path) as json_file:
            data = json.load(json_file)
            text = TextData.from_dict(data=data)

        text_intact = ""
        text_in = ""
        text_exp = ""

        for word_data in text.words:
            text_intact += word_data.word + word_data.punctuation
            non_alphanumeric_pattern = r'^[^\w"%]+$'
            if not re.match(non_alphanumeric_pattern, word_data.word):
                text_in += word_data.word
                text_exp += word_data.word
                # "-" should be also present?????? strange edge case
                if word_data.punctuation == "-" and not word_data.space_after:
                    text_exp += word_data.punctuation

            text_intact += " "
            if word_data.space_after or word_data.punctuation != "":
                text_in += " "
                text_exp += " "

        return text_in, text_exp, text_intact

    @staticmethod
    def _normalize_text_in(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[,!?.:;-]", " ", text)
        text = re.sub(r" +", " ", text)
        return text.strip()

    @staticmethod
    def _normalize_text_exp(text: str) -> str:
        text = text.lower()
        text = re.sub(r"\.\.\.", "…", text)
        text = re.sub(r"[!?.:;-]([^ ])", r" \1", text)
        text = re.sub(r"[…,!?.:;-]", " ", text)
        text = re.sub(r" +", " ", text)
        text = re.sub(r"…", "...", text)
        return text.strip()

    @staticmethod
    def _validate_equal_len(text_in: str, text_exp: str) -> None:
        """
        validate, whether text_in and text_exp are the same length after splitting

        :param text_in:
        :param text_exp:
        :return:
        """
        text_in_splitted = text_in.split(" ")
        text_exp_splitted = text_exp.split(" ")
        try:
            assert len(text_in_splitted) == len(text_exp_splitted)
        except AssertionError:
            print(
                len(text_in_splitted),
                len(text_exp_splitted),
                file=sys.stderr,
            )
            print(text_in_splitted, file=sys.stderr)
            print(text_exp_splitted, file=sys.stderr)
            for a, b in zip(text_in_splitted, text_exp_splitted):
                print(a, b, file=sys.stderr)
            print()


if __name__ == "__main__":
    converter = JsonToTsvConverter()
    converter.convert()
