import argparse
import sys
import logging
import json
from transformers import pipeline
import functools

from llm_utils import *
from qspan import find_supporting_quote
import re

# TODO: Plug in more representative examples.

SYSTEM_PROMPT = "You are a system that can split up composite questions and dependent questions, turning them into self-contained subquestions, particularly for the Dutch language."
EXAMPLES = [
    {'prompt': 'Sinds wanneer geldt deze maatregel en wat was destijds de motivatie?',
     'response': ['Sinds wanneer geldt deze maatregel?', 'Wat was destijds de motivatie voor deze maatregel?']},
    {'prompt': 'Heeft u de brief van de Indonesische overheid gelezen, en zoja, wat is uw reactie?',
     'response': ["Heeft u de brief van de Indonesische overheid gelezen", "Wat is uw reactie op de brief van de Indonesische overheid?"]},
    {'prompt': 'Wat is de grondwettelijke status van deze maatregel? Is dit onderzocht (en door wie)?',
     'response': ["Wat is de staatrechtelijke grondslag van deze maatregel?", "Is de staatrechtelijke grondslag van deze maatregel onderzocht?", "Door wie is de staatsrechtelijke grondslag van deze maatregel onderzocht?"]},
    {'prompt': 'Bent u bekend met het nieuwsbericht dat steeds meer asielzoekers via Luxemburg reizen? Zonee, waarom niet?',
     'response': ['Bent u bekend met het nieuwsbericht dat steeds meer asielzoekers via Luxemburg reizen?', 'Waarom bent u niet bekend met het nieuwsbericht dat steeds meer asielzoekers via Luxemburg reizen?']},
    {'prompt': 'Hoevaak en wanneer nemen mensen in Nederland de fiets? Wat is daarover uw mening?',
     'response': ["Hoevaak nemen mensen in Nederland de fiets?", "Wanneer nemen mensen in Nederland de fiets?", "Wat is uw mening over hoevaak en wanneer mensen in Nederland de fiets nemen?"]},
    {'prompt': 'Bent u het met mij eens dat dierenrecht een prominentere plek moet innemen in de samenleving? Zonee, waarom niet?',
     'response': ['Bent u het met mij eens dat dierenrecht een prominentere plek moet innemen in de samenleving?', 'Waarom vindt u niet dat dierenrecht een prominentere plek moet innemen in de samenleving?']},
]
for exe in EXAMPLES:
    exe['response'] = json.dumps(exe['response'])


# TODO: Include a 'raw' key in the output json? Pass along with the exception?!
# TODO: Add gradual temperature increase for retrying?!
# TODO: In case of splitandmerge, adapt the prompt so the model focuses on the last sentence? Will make it far more efficient, too.
# TODO: The ... doesn't work quite as it should, for discontinuous quotes... maybe |, or csq like "blabla", "blablabla"?
# TODO: Refactoring.

def main():

    logging.basicConfig(level=logging.INFO)

    argparser = argparse.ArgumentParser(description='Qsep')
    argparser.add_argument('file', nargs='?', type=argparse.FileType('r'), default=sys.stdin, help='Input file, one (composite) question per line; when omitted stdin.')
    argparser.add_argument('--model', nargs='?', default="unsloth/llama-3-70b-Instruct-bnb-4bit", type=str)
    argparser.add_argument('--list', action='store_true', help='Whether to give a json list with outputs per input, instead of potentially multiple lines per input.')
    argparser.add_argument('--json', action='store_true', help='Whether to give json output instead of plain strings. NB. Interacts with --validate; see README.')
    argparser.add_argument('--validate', action='store_true', help='Use LLM to link replies back to original quotes')

    argparser.add_argument('--temp', required=False, type=float, help='Temperature', default=.1)
    argparser.add_argument('--topp', required=False, type=float, help='Sample only from top probability', default=None)

    argparser.add_argument('--splitandmerge', required=False, type=int, default=None, help='Cuts question sequences into smaller chunks of n questions; recommended for longer sequences of questions, though really only with --validate enabled.')
    argparser.add_argument('--fuzzy', required=False, type=float, help='For retrieving quotations (if --validate), allow fuzzy matching, as a proportion of total characters.', default=0)
    argparser.add_argument('--retry', required=False, type=int, help='Max number of retries if response failed to parse.', default=5)
    argparser.add_argument('--validate_retry', required=False, type=int, help='Max number of retries if validation response failed to parse.', default=2)
    args = argparser.parse_args()

    if args.model == 'test':
        # TODO: Implement a fake language model for easier testing...
        args.model = 'llamafactory/tiny-random-Llama-3'

    if args.validate and not args.json:
        logging.warning("Are you sure you don't want --json output?")

    if args.splitandmerge and not args.validate:
        logging.warning("Are you sure you don't want --validate? Split and merge may result in a lot of duplicates otherwise!")

    pipe = functools.partial(pipeline("text-generation", model=args.model), max_new_tokens=1000, temperature=args.temp, top_p=args.topp)
    if not args.validate:
        parser = parse_json_or_itemized_list_of_strings
    else:
        # to be further instantiated per input line
        parser = get_validated_parser(pipe=pipe, validate_n_retries=args.validate_retry, fuzzy=args.fuzzy)

    for n, line in enumerate(args.file):

        if n > 0:
            print()

        line = line.strip()
        if not line:
            logging.warning(f'Empty line on input line {n}')
            if args.list:
                print(json.dumps([]))
            else:
                print()
            continue

        if args.splitandmerge is not None:
            # TODO: Refactor

            result = []

            for chunk_start, target_start, chunk_text in iter_question_tuples(line, args.splitandmerge):

                if args.validate:
                    parser = functools.partial(parser, original_text=chunk_text, char_offset=chunk_start, only_from_char=target_start)

                chat_start = make_chat_start(chunk_text, EXAMPLES, SYSTEM_PROMPT)
                try:
                    subresult = retry_until_parse(pipe, chat_start, parser, args.retry)
                except ValueError as e:
                    logging.warning(f'Failed parsing response for input chunk {n}.{chunk_start}; {e}')
                    continue

                if args.validate:
                    # for validate in the case of split and merge, don't keep the Nones, or we get too many duplicates:
                    subresult = [r for r in subresult if r['spans'] is not None]

                result.extend(subresult)

            if not some_succeeded:
                logging.warning(f'Failed parsing response for any tuple of input line {n}')
                continue

        else:
            if args.validate:
                parser = functools.partial(parser, original_text=line)

            chat_start = make_chat_start(line, EXAMPLES, SYSTEM_PROMPT)
            try:
                result = retry_until_parse(pipe, chat_start, parser, args.retry)
            except ValueError as e:
                logging.warning(f'Failed parsing response for input line {n}; {e}')
                continue

        # TODO: Refactor the various output formats
        if args.validate and not args.json:
            result = [res['rephrased'] for res in result]
        if args.list:
            if args.json:
                print(json.dumps(result))
            else:
                print(result) # TODO: Should be csv.
        else:
            for res in result:
                if args.json:
                    print(json.dumps(res))
                else:
                    print(res)


def iter_question_tuples(line: str, n_per_tuple: int):
    """
    >>> list(iter_question_tuples('Test? Hello? Not sure?', 1))
    [(0, 0, 'Test?'), (5, 5, ' Hello?'), (12, 12, ' Not sure?')]
    >>> list(iter_question_tuples('Test? Hello? Not sure?', 2))
    [(0, 0, 'Test?'), (0, 5, 'Test? Hello?'), (5, 12, ' Hello? Not sure?')]
    """

    questions = [None] * (n_per_tuple - 1) + list(re.finditer(r'[^?]+\?(?=(?: +[A-Z])|(?: *$))', line))
    tuples = zip(*[questions[n:] for n in range(n_per_tuple)])
    for questions_tuple in tuples:
        questions_tuple = tuple(filter(None, questions_tuple))
        chunk_start = questions_tuple[0].span()[0]
        target_start = questions_tuple[-1].span()[0]
        chunk_text = ''.join(match.group() for match in questions_tuple)

        yield chunk_start, target_start, chunk_text



def get_validated_parser(pipe, validate_n_retries, fuzzy):
    already_used = []

    def parser(raw, original_text, char_offset=0, only_from_char=0):
        results = []
        for rephrased in parse_json_or_itemized_list_of_strings(raw):
            spans = find_supporting_quote(original=original_text, rephrased=rephrased, pipe=pipe,
                                          n_retries=validate_n_retries, fail_ok=True, already_used=already_used,
                                          fuzzy=fuzzy, only_from_char=only_from_char - char_offset),
            if char_offset:
                for span in spans:
                    span['start'] += char_offset
                    span['end'] += char_offset

            result = {
                'spans': spans,
                'rephrased': rephrased,
            }
            results.append(result)

        return results

    return parser


def parse_json_or_itemized_list_of_strings(raw):
    try:
        return parse_json_list_of_strings(raw)
    except ValueError as e1:
        try:
            return parse_itemized_list_of_strings(raw)
        except ValueError as e2:
            raise ValueError(f'{e1}; {e2}')


def parse_json_list_of_strings(raw):
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError('Not a json string')
    if not isinstance(result, list):
        raise ValueError('Not a list')
    if any(not isinstance(x, str) for x in result):
        raise ValueError('List contains a non-string')
    return result


enum_regex = re.compile(r'[ \t]*\d+. +([^\n]+)')
item_regex = re.compile(r'[ \t]*- +([^\n]+)')

def parse_itemized_list_of_strings(raw):
    if len(result := enum_regex.findall(raw)) <= 1 and len(result := item_regex.findall(raw)) <= 1:
        raise ValueError('Not an itemized/enumerated list of strings')
    return [s.strip('"\'') for s in result]


if __name__ == '__main__':

    main()