import operator
import glob
import csv
import sys
from textblob import TextBlob

from typing import (
    Iterator,
    List
)

from core.models import (
    Token,
    Path
)


def system_slash():
    if sys.platform == 'win32':
        return '\\'
    return '/'

# ===== Replace term with dash   =====
def replace_term_phrase_with_dash(term):
    tmp = term.split(' ')
    new_term = "_".join(tmp)

    return new_term

# ===== Replace phrases with dash   =====
# TODO: figure out other phrases:
def replace_context_phrases_with_dash(term, context):
    blob = TextBlob(context)
    result = context
    result = result.replace("- ", "") # remove artifacts

    # replace term
    tmp = term.split(' ')
    new_term = "_".join(tmp)
    result.replace(term, new_term)

    # find and replace other phrases
    noun_phrases = blob.noun_phrases
    for phrase in noun_phrases:
        tmp_phrase_arr = phrase.split(' ')
        new_phrase = "_".join(tmp_phrase_arr)
        result = result.replace(phrase, new_phrase)

    return result

# ===== Remove stop words =====
def remove_stop_words(context):
    stop_words = ['is', 'a', 'A', 'will', 'be', 'for', 'the', 'on', 'to','in', 'of', '', 'and', 'by', 'that', 'an']
    results = ''
    tmp = context.split(' ')

    for stop_word in stop_words:
        if stop_word in tmp:
            tmp.remove(stop_word)
    results = " ".join(tmp)

    return results

#TODO: not all context are founded - the opened file is not reading till the end
def search_contexts_in_file(filename: str, token: Token) -> Token:
    contexts  = []
    with open(filename, mode='r') as datafile:
        reader = csv.reader(datafile, delimiter='.')

        try:
            for row in reader:
                #print("token.term: ", token.term)
                #print("row: ", row)

                if token.term in str(row):
                    #print("row: ", row)
                    clean_row = remove_stop_words(row[0])
                    clean_row = replace_context_phrases_with_dash(token.term, clean_row)
                    #print("clean_row: ", clean_row)
                    contexts.append(clean_row)

        except Exception as e:
              print("")
              #print("Exception read: ",e)

    #print("contexts: ", contexts)
    return Token(
        term=replace_term_phrase_with_dash(token.term), #token.term,
        value=token.value,
        convergence=token.convergence,
        contexts=contexts
    )

def read_file(filename: str, delimiter: str = ';') -> Iterator[Token]:
    with open(filename, mode='r') as terms:
        reader = csv.reader(terms, delimiter=delimiter)

        for row in reader:
            token, value = row

            yield Token(
                term=token,
                value=value
            )


def read_file_list(pattern: str, slash: str) -> Iterator[Path]:
    for path in glob.glob(pattern):
        parts = path.split(slash)

        if not parts:
            raise ValueError('Invalid file path')

        yield Path(
            path=path,
            name=parts[-1],
            order=''.join(filter(str.isdigit, parts[-1])),
        )


def load_list(path: str, folder: str, mask: str = '*.txt') -> List[Path]:
    slash = system_slash()

    pattern = '{path}{slash}{folder}{slash}{mask}'.format(
        path=path,
        slash=slash,
        folder=folder,
        mask=mask
    )

    files = read_file_list(
        pattern=pattern,
        slash=slash
    )
    return sorted(files, key=operator.attrgetter('order'))
