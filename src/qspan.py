import argparse
import sys
import logging
from transformers import pipeline
import functools
from llm_utils import *
import csv
import re


PROMPT_FORMAT = '> {original}\n\nGive an exact, literal quote (mind the punctuation!) from this passage that conveys exactly "{rephrase}", and no more.'

SYSTEM_PROMPT = "You are a system that can match paraphrases to the original quotations in the source text, specialized in questions, in particular for the Dutch language."

# TODO: Plugin more representative examples.

EXAMPLES = [
    {'original': 'Sinds wanneer geldt deze maatregel en wat was destijds de motivatie?',
     'rephrase': 'Wat was destijds de motivatie voor deze maatregel?',
     'response': 'wat was destijds de motivatie?'},
    {'original': 'Heeft u de brief van de Indonesische overheid gelezen, en zoja, wat is uw reactie?',
     'rephrase': 'Als u de brief van de Indonesische overheid gelezen heeft, wat is dan uw reactie?',
     'response': "zoja, wat is dan uw reactie?"},
    {'original': 'Bent u het met mij eens dat dierenrecht een prominentere plek moet innemen in de samenleving?',
     'rephrase': 'Vindt u ook dat dierenrecht een prominentere plek in de samenleving moet innemen?',
     'response': 'Bent u het met mij eens dat dierenrecht een prominentere plek moet innemen in de samenleving?'},
    {'original': 'Wat is de grondwettelijke status van deze maatregel? Is dit onderzocht?',
     'rephrase': 'Is de staatrechtelijke grondslag van deze maatregel onderzocht?',
     'response': "Is dit onderzocht?"},
    {'original': 'Hoevaak en wanneer nemen mensen in Nederland de fiets? Wat is daarover uw mening?',
     'rephrase': 'Hoevaak nemen mensen in Nederland de fiets?',
     'response': "Hoevaak ... nemen mensen in Nederland de fiets?"},
    {'original': 'Hoevaak en wanneer nemen mensen in Nederland de fiets? Wat is daarover uw mening?',
     'rephrase': 'Wat is uw mening over hoevaak en wanneer mensen in Nederland de fiets nemen?',
     'response': "Wat is daarover uw mening?"},
    {'original': "Wie doet dat en sinds wanneer",
     'rephrase': "Sinds wanneer wordt dat gedaan?",
     'response': "sinds wanneer?"},
]

for exe in EXAMPLES:
    exe['prompt'] = PROMPT_FORMAT.format(original=exe['original'], rephrase=exe['rephrase'])
    del exe['original']
    del exe['rephrase']


def main():

    logging.basicConfig(level=logging.INFO)

    argparser = argparse.ArgumentParser(description='Qsep')
    argparser.add_argument('file', nargs='?', type=argparse.FileType('r'), default=sys.stdin, help='Input file with pairs original,rephrased per line (csv); when omitted read from stdin.')
    argparser.add_argument('--model', nargs='?', default="unsloth/llama-3-70b-Instruct-bnb-4bit", type=str)
    argparser.add_argument('--json', action='store_true', help='Whether to give json output; otherwise each question on a new line, with empty line per input.')
    argparser.add_argument('--temp', required=False, type=float, help='Temperature', default=.1)
    argparser.add_argument('--topp', required=False, type=float, help='Sample only from top probability', default=None)
    argparser.add_argument('--retry', required=False, type=int, help='Max number of retries if response failed to parse.', default=5)
    args = argparser.parse_args()

    if args.model == 'test':
        args.model = 'llamafactory/tiny-random-Llama-3'

    pipe = functools.partial(pipeline("text-generation", model=args.model), max_new_tokens=1000, temperature=args.temp, top_p=args.topp)
    for n, (original, rephrased) in enumerate(csv.reader(args.file)):
        try:
            result = find_supporting_quote(original, rephrased, pipe, args.retry)
        except ValueError as e:
            logging.warning(f'Failed parsing response for input {n}; {e}')
            print()
        else:
            if args.json:
                print(json.dumps(result))
            else:
                for res in result:
                    print(res)
        print()


def find_supporting_quote(original, rephrased, pipe, n_retries, fail_ok=False):
    prompt = PROMPT_FORMAT.format(original=original, rephrase=rephrased)
    chat_start = make_chat_start(prompt, EXAMPLES, SYSTEM_PROMPT)
    return retry_until_parse(pipe,
                             chat_start,
                             parser=functools.partial(parse_string_quote_as_spans, original=original),
                             n_retries=n_retries,
                             fail_ok=fail_ok)


def parse_string_quote_as_spans(quote: str, original: str, ignore_punct=True, fuzzy=False) -> list[dict]:
    """
    >>> parse_string_quote_as_spans('de grote ... was lui', 'de grote grijze vos was lui')
    [{'start': 0, 'end': 8, 'text': 'de grote'}, {'start': 20, 'end': 27, 'text': 'was lui'}]
    """

    # TODO: Implement fuzzy=True
    if fuzzy:
        raise NotImplementedError

    quote_chunks = quote.split('...')

    clean_quote_chunks = [re.escape(chunk.strip()) for chunk in quote_chunks]
    regex_quote_chunks = [f'({chunk})' for chunk in clean_quote_chunks]
    regex = re.compile('.+'.join(regex_quote_chunks), flags=re.IGNORECASE)

    spans = []
    matches = list(regex.finditer(original))
    if len(matches) != 1 and ignore_punct:
        # TODO: refactor
        clean_quote_chunks_nopunct = [re.escape(chunk.strip().strip('.?!"\')([]')) for chunk in quote_chunks]
        regex_quote_chunks_nopunct = [f'({chunk})' for chunk in clean_quote_chunks_nopunct]
        regex_nopunct = re.compile('.+'.join(regex_quote_chunks_nopunct), flags=re.IGNORECASE)
        matches = list(regex_nopunct.finditer(original))
    if len(matches) != 1:
        raise ValueError(f'{quote=}')

    match = matches[0]
    for n in range(1, len(regex_quote_chunks)+1):
        start, end = match.span(n)
        spans.append({'start': start, 'end': end, 'text': match.group(n)})

    return spans


if __name__ == '__main__':

    main()