import operator
import glob
import csv
import sys
from textblob import TextBlob
#import io
#import re

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


def restore_windows_1252_characters(s):
    """Replace C1 control characters in the Unicode string s by the
    characters at the corresponding code points in Windows-1252,
    where possible.

    """
    import re
    def to_windows_1252(match):
        try:
            return bytes([ord(match.group(0))]).decode('windows-1252')
        except UnicodeDecodeError:
            # No character at the corresponding code point: remove it.
            return ''
    return re.sub(r'[\u0080-\u0099]', to_windows_1252, s)

# def remove_annoying_characters(text):
# chars = {
#     '\xc2\x82' : ',',        # High code comma
#     '\xc2\x84' : ',,',       # High code double comma
#     '\xc2\x85' : '...',      # Tripple dot
#     '\xc2\x88' : '^',        # High carat
#     '\xc2\x91' : '\x27',     # Forward single quote
#     '\xc2\x92' : '\x27',     # Reverse single quote
#     '\xc2\x93' : '\x22',     # Forward double quote
#     '\xc2\x94' : '\x22',     # Reverse double quote
#     '\xc2\x95' : ' ',
#     '\xc2\x96' : '-',        # High hyphen
#     '\xc2\x97' : '--',       # Double hyphen
#     '\xc2\x99' : ' ',
#     '\xc2\xa0' : ' ',
#     '\xc2\xa6' : '|',        # Split vertical bar
#     '\xc2\xab' : '<<',       # Double less than
#     '\xc2\xbb' : '>>',       # Double greater than
#     '\xc2\xbc' : '1/4',      # one quarter
#     '\xc2\xbd' : '1/2',      # one half
#     '\xc2\xbe' : '3/4',      # three quarters
#     '\xca\xbf' : '\x27',     # c-single quote
#     '\xcc\xa8' : '',         # modifier - under curve
#     '\xcc\xb1' : ''          # modifier - under line
# }
# def replace_chars(match):
#     char = match.group(0)
#     return chars[char]
# return re.sub('(' + '|'.join(chars.keys()) + ')', replace_chars, text)

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
    stop_words = ['is', 'a', 'A', 'will', 'be', 'for', 'the', 'on', 'to','in', 'of', '', 'and', 'by', 'that', 'an', 'are']
    results = ''
    tmp = context.split(' ')

    #print("start remove_stop_words: ", tmp)
    for stop_word in stop_words:
        if stop_word in tmp:
            tmp.remove(stop_word)
    results = " ".join(tmp)

    #print("end remove_stop_words: ", results)
    return results

#TODO: not all context are founded - the opened file is not reading till the end
def search_contexts_in_file(filename: str, token: Token) -> Token:
    contexts  = []
    #print("Default buffer size:",io.DEFAULT_BUFFER_SIZE)

    file=open(filename, mode="r", buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None)
    print("line_buffering: ",file.line_buffering)
    file_contents=file.buffer
    for line in file_contents:
        #print("before file_contents line: ",line)
        row = line.rstrip().decode('utf-8') #.decode('windows-1252')
        #row = re.sub(r'[\xc2\x99]'," ", row)
        if token.term in row:
            #clean_row = remove_annoying_characters(row)
            clean_row = remove_stop_words(row)
            clean_row = replace_context_phrases_with_dash(token.term, clean_row)
            contexts.append(clean_row)

    file.close()
        # with open(filename, mode='r', buffering=io.DEFAULT_BUFFER_SIZE * 3) as datafile:
        #     reader = csv.reader(datafile, delimiter='.')
		#
        #     try:
        #         for row in reader:
        #             #print("token.term: ", token.term)
        #             print("row: ", row)
		#
        #             if token.term in str(row):
        #                 #print("row: ", row)
        #                 clean_row = remove_stop_words(row[0])
        #                 clean_row = replace_context_phrases_with_dash(token.term, clean_row)
        #                 #print("clean_row: ", clean_row)
        #                 contexts.append(clean_row)
		#
        #     except Exception as e:
        #           #print("")
        #           print("Exception read: ",e)
		#
        # datafile.close()

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
