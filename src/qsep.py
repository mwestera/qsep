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

SYSTEM_PROMPT = "You are a system that can break down a potentially complex question and dependent questions into self-contained subquestions, particularly for the Dutch language."
EXAMPLES = [
    {'prompt': 'Sinds wanneer geldt deze maatregel en wat was destijds de motivatie?',
     'response': ['Sinds wanneer geldt deze maatregel?', 'Wat was destijds de motivatie voor deze maatregel?']},
    {'prompt': 'Heeft u de brief van de Indonesische overheid gelezen, en zoja, wat is uw reactie?',
     'response': ["Heeft u de brief van de Indonesische overheid gelezen", "Wat is uw reactie op de brief van de Indonesische overheid?"]},
    {'prompt': 'Bent u het met mij eens dat dierenrecht een prominentere plek moet innemen in de samenleving?',
     'response': ['Bent u het met mij eens dat dierenrecht een prominentere plek moet innemen in de samenleving?']},
    {'prompt': 'Wat is de grondwettelijke status van deze maatregel? Is dit onderzocht (en door wie)?',
     'response': ["Wat is de staatrechtelijke grondslag van deze maatregel?", "Is de staatrechtelijke grondslag van deze maatregel onderzocht?", "Door wie is de staatsrechtelijke grondslag van deze maatregel onderzocht?"]},
    {'prompt': 'Bent u bekend met het nieuwsbericht dat steeds meer asielzoekers via Luxemburg reizen?',
     'response': ['Bent u bekend met het nieuwsbericht dat steeds meer asielzoekers via Luxemburg reizen?']},
    {'prompt': 'Hoevaak en wanneer nemen mensen in Nederland de fiets? Wat is daarover uw mening?',
     'response': ["Hoevaak nemen mensen in Nederland de fiets?", "Wanneer nemen mensen in Nederland de fiets?", "Wat is uw mening over hoevaak en wanneer mensen in Nederland de fiets nemen?"]},
]
for exe in EXAMPLES:
    exe['response'] = json.dumps(exe['response'])


def main():

    logging.basicConfig(level=logging.INFO)

    argparser = argparse.ArgumentParser(description='Qsep')
    argparser.add_argument('file', nargs='?', type=argparse.FileType('r'), default=sys.stdin, help='Input file, one (composite) question per line; when omitted stdin.')
    argparser.add_argument('--model', nargs='?', default="unsloth/llama-3-70b-Instruct-bnb-4bit", type=str)
    argparser.add_argument('--list', action='store_true', help='Whether to give a json list with outputs per input, instead of potentially multiple lines per input.')
    argparser.add_argument('--json', action='store_true', help='Whether to give json output instead of plain strings. NB. Interacts with --validate; see README.')
    argparser.add_argument('--temp', required=False, type=float, help='Temperature', default=.1)
    argparser.add_argument('--topp', required=False, type=float, help='Sample only from top probability', default=None)
    argparser.add_argument('--validate', action='store_true', help='Use LLM to link replies back to original quotes')
    argparser.add_argument('--retry', required=False, type=int, help='Max number of retries if response failed to parse.', default=5)
    args = argparser.parse_args()

    if args.model == 'test':
        args.model = 'llamafactory/tiny-random-Llama-3'

    if args.validate and not args.json:
        logging.warning("Are you sure you don't want --json output?")

    pipe = functools.partial(pipeline("text-generation", model=args.model), max_new_tokens=1000, temperature=args.temp, top_p=args.topp)
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

        chat_start = make_chat_start(line, EXAMPLES, SYSTEM_PROMPT)
        if not args.validate:
            parser = parse_json_or_itemized_list_of_strings
        else:
            def parser(raw):
                return [{'spans': find_supporting_quote(original=line, rephrased=rephrased, pipe=pipe, n_retries=args.retry, fail_ok=True),
                         'rephrased': rephrased} for rephrased in parse_json_list_of_strings(raw)]

        # TODO: Refactor the various output formats
        try:
            result = retry_until_parse(pipe, chat_start, parser, args.retry)
        except ValueError as e:
            logging.warning(f'Failed parsing response for input line {n}; {e}')
        else:
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