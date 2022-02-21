"""
Functions used to determine if a context has an answer.
Currently supported:
- has_answer_field: return ctx['has_answer'] set during retrieval
- string_match: check whether the answer appears in ctx['text']
- normalized_title: check whether ctx['title'] matches an answer
- regex: Use regular expressions to match ctx['text'] with answer patterns
"""
import re
import unicodedata
from utils.tokenizers import SimpleTokenizer


def normalize(text):
    return unicodedata.normalize('NFD', text)


###############################################################################
### HAS_ANSWER FUNCTIONS   ####################################################
###############################################################################
def has_answer_field(ctx, answer_lst, tokenizer=None):
    return ctx['has_answer']

# True iff the text, including the title, includes an answer
def string_title_match(ctx, answer_lst, tokenizer=None):
    tokenizer = DEFAULT_TOKENIZER if tokenizer is None else tokenizer
    corpus = f"{ctx['title']} {ctx['text']}"
    text = tokenizer.tokenize(corpus).words(uncased=True)

    for answer in answer_lst:
        answer = normalize(answer)
        answer = tokenizer.tokenize(answer)
        answer = answer.words(uncased=True)

        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False


# True iff the text, excluding the title, includes an answer.
def string_match(ctx, answer_lst, tokenizer=None):
    tokenizer = DEFAULT_TOKENIZER if tokenizer is None else tokenizer
    text = tokenizer.tokenize(ctx['text']).words(uncased=True)

    for answer in answer_lst:
        answer = normalize(answer)
        answer = tokenizer.tokenize(answer)
        answer = answer.words(uncased=True)

        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False


# True iff the title starts with an answer.
def normalized_title(ctx, answer_lst, tokenizer=None):
    for answer in answer_lst:
        answer = answer.lower().strip()
        title = ctx['title'].lower().strip()
        if title.startswith(answer):
            return True
    return False


# True iff the text contains at least one regex match with an answer.
def regex(ctx, answer_lst, tokenizer=None):
    text = ctx['text']
    for answer in answer_lst:
        answer = normalize(answer)
        if regex_match(text, answer):
            return True
    return False


def regex_match(ctx, answer, tokenizer=None):
    """Test if a regex pattern is contained within a text."""
    text = ctx['text']
    try:
        pattern = re.compile(answer, flags=re.IGNORECASE + re.UNICODE + re.MULTILINE)
    except BaseException:
        return False
    return pattern.search(text) is not None


DEFAULT_TOKENIZER = SimpleTokenizer()
HAS_ANS_FNS = {
    'has_answer': has_answer_field,
    'regex': regex,
    'string': string_match,
    'title': normalized_title,
    'string_title': string_title_match,
}
