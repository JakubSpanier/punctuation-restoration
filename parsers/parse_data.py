import glob
import os.path
import random
from argparse import ArgumentParser
from parse_json2 import load_json


class JsonToConllConverter:
    def __init__(self, train_path, test_path, data_paths, save_path):
        self.train_path = train_path
        self.test_path = test_path
        self.data_paths = data_paths
        self.save_path = save_path

    def _read_names(self, path):
        names = []
        with open(path) as file:
            for line in file:
                line = line.strip()
                name, _ = line.split("\t")
                names.append(name)
        return names

    def _get_json_paths(self):
        train_names = self._read_names(self.train_path)
        test_names = self._read_names(self.test_path)

        train_paths, test_paths, rest_paths = [], [], []
        for path in self.data_paths:
            json_paths = glob.glob(os.path.join(path, "*.json"))
            for json_path in json_paths:
                basename = os.path.basename(json_path).split(".")[0]
                if basename in train_names:
                    train_paths.append(json_path)
                elif basename in test_names:
                    test_paths.append(json_path)
                else:
                    rest_paths.append(json_path)

        train_paths = self._sort_paths(train_paths, train_names)
        test_paths = self._sort_paths(test_paths, test_names)

        random.seed(0)
        random.shuffle(rest_paths)

        return train_paths, test_paths, rest_paths

    def _sort_paths(self, paths, names):
        sorted_paths = []
        for name in names:
            for path in paths:
                if name in path:
                    sorted_paths.append(path)
                    break
        return sorted_paths

    def _write_output(self, name, paths):
        with open(os.path.join(self.save_path, f"{name}_expected.tsv"), "w") as out_expected, open(
            os.path.join(self.save_path, f"{name}_in.tsv"), "w"
        ) as out_in:
            for path in paths:
                json_in, json_expected = load_json(path)
                basename = os.path.basename(path).split(".")[0]
                out_in.write(f"{basename}\t{json_in}\n")
                out_expected.write(f"{json_expected}\n")

    def convert(self):
        train_paths, test_paths, rest_paths = self._get_json_paths()

        for name, paths in [("test", test_paths), ("train", train_paths), ("rest", rest_paths)]:
            self._write_output(name, paths)

        print(len(test_paths), len(test_paths), len(train_paths), len(train_paths))


if __name__ == "__main__":
    parser = ArgumentParser(description="Convert JSON to CONLL")
    parser.add_argument("--train_path", default="2021-punctuation-restoration/train/in.tsv", help="Path to train in.tsv")
    parser.add_argument("--test_path", default="2021-punctuation-restoration/test-A/in.tsv", help="Path to test in.tsv")
    parser.add_argument("data", nargs="+", help="Paths to dirs with JSONs")
    parser.add_argument("--save_path", default=".", help="Path to directory")
    args = parser.parse_args()

    converter = JsonToConllConverter(args.train_path, args.test_path, args.data, args.save_path)
    converter.convert()
