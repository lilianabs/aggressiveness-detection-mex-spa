import re
import string
import spacy

nlp = spacy.load("es_core_news_sm")


def lowercase(text: str) -> str:
    return text.lower()


def remove_special_tokens(text: str) -> str:
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\<URL\>", "", text)
    text = re.sub(r"\n", " ", text)
    return text


def remove_punctuation(text: str) -> str:
    spanish_punctuation = string.punctuation + "¿¡"
    table = str.maketrans("", "", spanish_punctuation)
    return text.translate(table)


def remove_stopwords(text: str) -> str:
    doc = nlp(text)
    return " ".join(token.text for token in doc if not token.is_stop)


def remove_extra_whitespace(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = text.lstrip()
    text = text.rstrip()
    return text

_STEPS = {
    "remove_special_tokens": remove_special_tokens,
    "remove_stopwords": remove_stopwords,
    "remove_punctuation": remove_punctuation,
    "lowercase": lowercase,
    "remove_extra_whitespace": remove_extra_whitespace,
}


def clean_text(text: str, steps: list[str]) -> str:
    for step in steps:
        text = _STEPS[step](text)
    return text
