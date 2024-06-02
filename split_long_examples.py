import pandas as pd
from transformers import AutoTokenizer
from pathlib import Path
from typing import Optional


class DataSplitter:
    def __init__(
        self,
        data_file: Path,
        tokenizer_path: str,
        max_seq_len: int = 256,
        stride: float = 1.0,
        out_file: Optional[Path] = None,
    ):
        """
        Initializes the DataSplitter with the provided parameters.

        :param data_file: Path to the input .tsv data file.
        :param tokenizer_path: Path or name of the tokenizer to use.
        :param max_seq_len: Maximum sequence length for tokenized words.
        :param stride: Stride to use when splitting long examples.
        :param out_file: Path to the output file.
        """
        self.data_path = data_file
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.max_seq_len = max_seq_len
        self.stride = stride
        self.out_file = out_file or data_file.with_name(
            f"{data_file.stem}_splitted.tsv"
        )

    def _split_long_examples(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Splits long examples into smaller chunks based on the maximum sequence length.

        :param data: The input DataFrame containing sentences to split.
        :return: A DataFrame with the split sentences.
        """
        splitted_data = []

        for sentence_id, example in data.groupby("sentence_id"):
            words_with_labels = []
            words_in_example = 0
            tokenized_len = 0
            token_lens = []
            chunk_id = 0

            for word, label, space in zip(example.words, example.labels, example.times):
                tokenized_word = self.tokenizer.tokenize(word)
                if tokenized_len + len(tokenized_word) >= self.max_seq_len - 1:
                    splitted_data.extend(
                        [
                            (w, l, s, f"{sentence_id}_{chunk_id}")
                            for w, l, s in words_with_labels
                        ]
                    )
                    chunk_id += 1
                    offset = int(words_in_example * self.stride)
                    words_with_labels = words_with_labels[offset:]
                    tokenized_len -= sum(token_lens[:offset])
                    token_lens = token_lens[offset:]
                    words_in_example -= offset

                token_lens.append(len(tokenized_word))
                tokenized_len += len(tokenized_word)
                words_with_labels.append((word, label, space))
                words_in_example += 1

            if tokenized_len >= 0:
                splitted_data.extend(
                    [
                        (w, l, s, f"{sentence_id}_{chunk_id}")
                        for w, l, s in words_with_labels
                    ]
                )

        return pd.DataFrame(
            splitted_data, columns=["words", "labels", "times", "sentence_id"]
        )

    def run(self):
        """
        Executes the splitting process and saves the output to the specified file.
        """
        data = pd.read_csv(
            self.data_path,
            sep="\t",
            keep_default_na=False,
            dtype={"words": "str", "labels": "str", "times": "str"},
        )

        splitted_data = self._split_long_examples(data)
        splitted_data.to_csv(self.out_file, sep="\t", index=False)


# Example usage
if __name__ == "__main__":
    tokenizer_path = "allegro/herbert-base-cased"
    original_train_path = lambda name: Path(
        f"parsed_data/original_train.tsv_{name}_1353.tsv"
    )
    original_test_path = lambda name: Path(
        f"parsed_data/original_test-A.tsv_{name}_1353.tsv"
    )
    rest_path = lambda name: Path(f"parsed_data/text.rest/rest.tsv_{name}_1353.tsv")
    names = ("dev", "train", "test")
    path_functions = (original_train_path, original_test_path, rest_path)
    files = [
        (path_function(name), path_function(name).with_suffix(".tsv.s"))
        for name in names
        for path_function in path_functions
    ]

    for data_file, out_file in files:
        splitter = DataSplitter(
            data_file=data_file,
            tokenizer_path=tokenizer_path,
            max_seq_len=256,
            stride=1.0,
            out_file=out_file,
        )
        splitter.run()
