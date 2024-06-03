from enum import Enum


class Label(str, Enum):
    NO_PUNCTUATION = "B"
    COLON = ":"
    SEMICOLON = ";"
    COMMA = ","
    PERIOD = "."
    HYPHEN = "-"
    ELLIPSIS = "..."
    QUESTION = "?"
    EXCLAMATION = "!"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))

    @classmethod
    def only_punctuations(cls):
        return [punctuation for punctuation in cls if punctuation != cls.NO_PUNCTUATION]
