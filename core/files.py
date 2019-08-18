import operator
import glob
import csv
import sys

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


def search_contexts_in_file(filename: str, token: Token) -> Token:
    contexts  = []
    with open(filename, mode='r') as datafile:
        reader = csv.reader(datafile, delimiter='.')

        try:
            for row in reader:
                #print("row: ", row)
                if token.term in str(row):
                    #print("row: ", row)
                    contexts.append(row)

        except Exception as e:
              print("")
              #print("Exception read: ",e)

    #print("contexts: ", contexts)
    yield Token(
        term=token.term,
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
