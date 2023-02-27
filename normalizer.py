import re
import unicodedata

# non-ASCII letters that are not separated by "NFKD" normalization
ADDITIONAL_DIACRITICS = {
    "œ": "oe",
    "Œ": "OE",
    "ø": "o",
    "Ø": "O",
    "æ": "ae",
    "Æ": "AE",
    "ß": "ss",
    "ẞ": "SS",
    "đ": "d",
    "Đ": "D",
    "ð": "d",
    "Ð": "D",
    "þ": "th",
    "Þ": "th",
    "ł": "l",
    "Ł": "L",
}


def remove_symbols_and_diacritics(s: str, keep=""):
    """
    Replace any other markers, symbols, and punctuations with a space,
    and drop any diacritics (category 'Mn' and some manual mappings)
    """
    return "".join(
        c
        if c in keep
        else ADDITIONAL_DIACRITICS[c]
        if c in ADDITIONAL_DIACRITICS
        else ""
        if unicodedata.category(c) == "Mn"
        else " "
        if unicodedata.category(c)[0] in "MSP"
        else c
        for c in unicodedata.normalize("NFKD", s)
    )


class TextNormalizer:
    def __init__(self):
        self.clean = remove_symbols_and_diacritics

    def __call__(self, s: str):
        s = s.lower()
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        s = s.replace("\n", " ")  # remove new line
        s = s.replace("\xa0", "")  # remove non-breaking space
        s = re.sub(r"mmm|euh", "", s)  # remove fillers
        s = self.clean(s).lower()
        s = re.sub(r"\s+", " ", s)  # replace one or more whitespace with only one
        s = re.sub(
            r"(\w)(\1{2,})", r"\1", s
        )  # replace prolonged words with standard spelling
        s = re.sub(r"\b(.+)(\b\1\b)+", r"\1", s)  # remove repeated phrases

        return s
