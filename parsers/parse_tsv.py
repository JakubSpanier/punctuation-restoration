import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class TsvParser:
    def __init__(
        self,
        in_path: Optional[Path] = None,
        expected_path: Optional[Path] = None,
        times_dir: Optional[Path] = None,
        save_path: Optional[Path] = None
    ) -> None:
        """
        :param in_path: Path to in.tsv
        :param expected_path: Path to expected.tsv
        :param times_dir: Path to dir with *.clntmstmp, basically the forced aligned texts
        :param save_path: Path to save output
        """
        self.in_path = in_path if in_path else Path("data/train/in.tsv")
        self.expected_path = expected_path if expected_path else Path("data/train/expected.tsv")
        self.times_dir = times_dir
        self.save_path = save_path if save_path else Path("parsed_data/original_train.conll")

    def convert(self):
        """ Convert .tsv files to .conll """
        with open(self.save_path, "w") as out_file:
            for in_line, expected_line in zip(open(self.in_path), open(self.expected_path)):
                in_line = in_line.strip()
                expected_line = expected_line.strip()
                wikipunct_text_id, text = in_line.split("\t")

                try:
                    assert len(text.split(" ")) == len(expected_line.split(" "))
                except AssertionError:
                    logger.warning("Source text and expected text differ!")
                    continue

                if self.times_dir:
                    times = self._times_after_tokens(f"{self.times_dir}/{wikipunct_text_id}.clntmstmp")
                    matched = self._match_times(times, expected_line.split(" "))

                for i, (in_token, expected_token) in enumerate(zip(text.split(" "), expected_line.split(" "))):
                    expected_token = expected_token.lower()
                    label = self._determine_label(in_token, expected_token)

                    if self.times_dir:
                        out_file.write(f"{in_token}\t{label}\t{matched[i]}\n")
                    else:
                        out_file.write(f"{in_token}\t{label}\n")
                out_file.write("\n")

    @staticmethod
    def _read_times_data(path):
        data = []
        with open(path) as file:
            for line in file:
                line = line.strip()
                if line == "</s>":
                    continue
                times, text = line.split(" ")
                start, end = map(int, times[1:-1].split(","))
                data.append((start, end, text))
        return data

    def _times_after_tokens(self, path):
        data = self._read_times_data(path)
        result = []
        for i in range(len(data)):
            start, end, text = data[i]
            przerwa = data[i + 1][0] - end if i + 1 < len(data) else 0
            result.append((text, przerwa))
        return result

    @staticmethod
    def _match_times(times, expected):
        matched = []
        times_text = ""
        times_indexes = {}

        for token, time in times:
            times_indexes[len(times_text)] = time
            times_text += token.lower()

        index = 0
        for token in expected:
            found_index = times_text.find(token.lower(), index)
            if found_index >= 0:
                if found_index in times_indexes:
                    matched.append(times_indexes[found_index])
                    index = found_index
                else:
                    matched.append(-1)
            else:
                matched.append(-1)
        return matched

    @staticmethod
    def _determine_label(in_token, expected_token):
        if in_token == expected_token:
            return "B"
        elif in_token == expected_token[:-1]:
            return expected_token[-1]
        elif in_token == expected_token[:-3] and expected_token[-3:] == "...":
            return expected_token[-3:]
        else:
            print("ERROR", in_token, expected_token, file=sys.stderr)
            return "B"


if __name__ == "__main__":
    tsv_parser = TsvParser()
    tsv_parser.convert()
