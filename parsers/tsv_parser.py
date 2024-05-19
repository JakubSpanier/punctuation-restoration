import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class TsvParser:
    def __init__(
        self,
        in_path: Optional[Path] = None,
        expected_path: Optional[Path] = None,
        fa_transcriptions_directory: Optional[Path] = None,
        save_path: Optional[Path] = None,
    ) -> None:
        """
        :param in_path: Path to in.tsv
        :param expected_path: Path to expected.tsv
        :param fa_transcriptions_directory: Path to dir with *.clntmstmp, basically the forced aligned texts
        :param save_path: Path to save output
        """
        # a or b is just more pythonic way to do a if a else b
        self.in_path = in_path or Path("data/train/in.tsv")
        self.expected_path = expected_path or Path("data/train/expected.tsv")
        # not sure which path should be default
        # Path("data/fa/poleval_fa.train") or Path("data/fa/poleval_fa.train.with_punctuation") 🤔
        self.fa_transcriptions_directory = fa_transcriptions_directory
        self.save_path = save_path or Path("parsed_data/original_train.conll")

    def convert(self):
        """Convert .tsv files to .conll"""
        with open(self.save_path, "w") as out_file:
            for in_line, expected_line in zip(
                open(self.in_path), open(self.expected_path)
            ):
                in_line = in_line.strip()
                expected_line = expected_line.strip()
                wikipunct_text_id, in_text = in_line.split("\t")

                try:
                    assert len(in_text.split(" ")) == len(expected_line.split(" "))
                except AssertionError:
                    logger.warning("Source text and expected text differ!")
                    continue

                if self.fa_transcriptions_directory:
                    words_and_breaks = self._calculate_breaks(
                        Path(
                            f"{self.fa_transcriptions_directory}/{wikipunct_text_id}.clntmstmp"
                        )
                    )
                    matched = self._provide_breaks_for_expected(
                        words_and_breaks, expected_line.split(" ")
                    )

                for i, (in_token, expected_token) in enumerate(
                    zip(in_text.split(" "), expected_line.split(" "))
                ):
                    expected_token = expected_token.lower()
                    label = self._determine_label(in_token, expected_token)

                    if self.fa_transcriptions_directory:
                        out_file.write(f"{in_token}\t{label}\t{matched[i]}\n")
                    else:
                        out_file.write(f"{in_token}\t{label}\n")
                out_file.write("\n")

    @staticmethod
    def _parse_transcript(path: Path) -> list[tuple[int, int, str]]:
        """
        parse text from forced-aligned texts
        (690,750) we
        (840,1350) wrocławiu
        (1650,1920) walkę
        ...
        </s>

        Returns:
            [(690, 750, "we"), (840, 1350, "wrocławiu"), (1650, 1920, "walkę"), ...]
        """
        data = []
        with open(path) as file:
            for line in file:
                line = line.strip()
                if line == "</s>":
                    continue
                timestamps, word = line.split(" ")
                # timestamps = "(1290,1320)"
                start, end = map(int, timestamps.strip("()").split(","))
                data.append((start, end, word))
        return data

    def _calculate_breaks(self, path: Path) -> list[tuple[str, int]]:
        """
        calculate breaks between end timestamp and start timestamp of the next word

        Returns:
            [("we", 90), ("wrocławiu", 300), ...]
        """
        result = []
        data = self._parse_transcript(path)
        for i, (_start, end, word) in enumerate(data[:-1]):
            break_ = data[i + 1][0] - end
            result.append((word, break_))
        # add the last word, and the break (so basically 0)
        result.append((data[-1][2], 0))

        return result

    @staticmethod
    def _provide_breaks_for_expected(
        words_and_breaks: list[tuple[str, int]], expected_text: list[str]
    ) -> list[int]:
        """
        calculate the break times for expected text
        if the word matches the word from fa transcript, it returns the break to the next word
        otherwise it returns -1

        :param words_and_breaks:
        :param expected_text:
        :return:

        Returns:
            [90, 300, -1, 210, -1, ...]
        """
        matched = []
        times_text = ""
        times_indexes = {}

        for word, break_ in words_and_breaks:
            times_indexes[len(times_text)] = break_
            times_text += word.lower()

        index = 0
        for word in expected_text:
            found_index = times_text.find(word.lower(), index)
            if found_index < 0:  # not found
                matched.append(-1)
                continue

            if found_index in times_indexes:
                matched.append(times_indexes[found_index])
                index = found_index
            else:
                matched.append(-1)

        return matched

    @staticmethod
    def _determine_label(in_token: str, expected_token: str) -> str:
        """
        if the tokens are equal, returns B
        otherwise check, if they are almost equal and
        expected word contains punctuation

        :param in_token:
        :param expected_token:
        :returns
        """
        if in_token == expected_token:
            return "B"
        elif in_token == expected_token[:-1]:
            return expected_token[-1]
        elif in_token == expected_token[:-3] and expected_token[-3:] == "...":
            return "..."
        else:
            logger.error(f"Missmatch! Words aren't equal, ({in_token=}!={expected_token=})")
            return "B"


if __name__ == "__main__":
    tsv_parser = TsvParser()
    tsv_parser.convert()
