import argparse
import sys
import logging
from transformers import pipeline
import functools
from llm_utils import *
import csv
import regex


PROMPT_FORMAT = '> {original}\n\nGive an exact, literal quote from this passage that conveys the same intent as "{rephrase}", and no more.'

SYSTEM_PROMPT = "You are a system that can match paraphrases to the original quotations in the source text, specialized in questions, in particular for the Dutch language."

# TODO: Plugin more representative examples; more examples with ...

EXAMPLES = [
    {'original': 'Sinds wanneer geldt deze maatregel en wat was destijds de motivatie (is deze openbaar)?',
     'rephrase': 'Wat was destijds de motivatie voor deze maatregel?',
     'response': 'wat was destijds de motivatie'},
    {'original': 'Heeft u de brief van de Indonesische overheid gelezen, en zoja, wat is uw reactie?',
     'rephrase': 'Wat is uw reactie op de brief van de Indonesische overheid?',
     'response': "wat is uw reactie?"},
    {'original': 'Bent u het met mij eens dat dierenrecht en milieubescherming een prominentere plek moeten innemen in de samenleving?',
     'rephrase': 'Vindt u ook dat milieubescherming een prominentere plek in de samenleving moet innemen?',
     'response': 'Bent u het met mij eens dat ... milieubescherming een prominentere plek moeten innemen in de samenleving?'},
    {'original': 'Wat is de grondwettelijke status van deze maatregel, is dit onderzocht, en door wie?',
     'rephrase': 'Is de staatrechtelijke grondslag van deze maatregel onderzocht?',
     'response': "is dit onderzocht"},
    {'original': 'Hoevaak en wanneer nemen mensen in Nederland de fiets? Wat is daarover uw mening?',
     'rephrase': 'Hoevaak nemen mensen in Nederland de fiets?',
     'response': "Hoevaak ... nemen mensen in Nederland de fiets?"},
    {'original': 'Hoevaak en wanneer nemen mensen in Nederland de fiets, wat is daarover (in het kader van de volksgezondheid) uw mening en is die openbaar?',
     'rephrase': 'Wat is uw mening over hoevaak en wanneer mensen in Nederland de fiets nemen?',
     'response': "wat is daarover uw mening"},
    {'original': "Sinds wanneer wordt dat gedaan en door wie?",
     'rephrase': "Sinds wanneer wordt dat gedaan?",
     'response': "Sinds wanneer wordt dat gedaan"},
    {'original': "Aan de hand van welke regelgeving en door welke commissie is het besluit omtrent artikel 27 genomen, en is daar voldoende inspraak bij geweest?",
     'rephrase': "Aan de hand van welke regelgeving is het besluit omtrent artikel 27 genomen?",
     'response': "Aan de hand van welke regelgeving ... is het besluit omtrent artikel 27 genomen"},
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
    argparser.add_argument('--fuzzy', required=False, type=float, help='For retrieving quotations, allow fuzzy matching, as a proportion of total characters.', default=0.0)
    argparser.add_argument('--retry', required=False, type=int, help='Max number of retries if response failed to parse.', default=5)
    args = argparser.parse_args()

    if args.model == 'test':
        args.model = 'llamafactory/tiny-random-Llama-3'

    pipe = functools.partial(pipeline("text-generation", model=args.model), max_new_tokens=1000, temperature=args.temp, top_p=args.topp)
    for n, (original, rephrased) in enumerate(csv.reader(args.file)):
        try:
            result = find_supporting_quote(original, rephrased, pipe, n_retries=args.retry, fuzzy=args.fuzzy)
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


def find_supporting_quote(original: str, rephrased: str, pipe, n_retries: int, fail_ok=False, already_used=None, fuzzy=0.0):
    prompt = PROMPT_FORMAT.format(original=original, rephrase=rephrased)
    chat_start = make_chat_start(prompt, EXAMPLES, SYSTEM_PROMPT)
    return retry_until_parse(pipe,
                             chat_start,
                             parser=functools.partial(parse_string_quote_as_spans, original=original, already_used=already_used, fuzzy=fuzzy),
                             n_retries=n_retries,
                             fail_ok=fail_ok)


# # Currently disabled because I don't think LLMs can count characters, and handling discontinuous spans not yet attempted.
#
# def parse_json_quote_as_spans(quote: str, original: str, fuzzy=0.0, already_used=None) -> list[dict]:
#     try:
#         d = json.loads(quote)
#     except json.JSONDecodeError as e:
#         raise ValueError(f"No valid JSON in {quote}")
#
#     if 'start' not in d or 'end' not in d or 'text' not in d:
#         raise ValueError(f"Key missing in {quote}")
#
#     start, end, text = d['start'], d['end'], d['text']
#     if not (0 <= start <= end <= len(original)):
#         raise ValueError(f"No proper start/end indices in {quote}")
#
#     target = original[start:end]
#     if target != text:
#         raise ValueError(f"Start:end results in different string ({text} != {target}) for {quote}")
#
#     parse_string_quote_as_spans(text, original, fuzzy, already_used)
#


# TODO: Implement in-dialogue-retrying with feedback


def parse_string_quote_as_spans(quote: str, original: str, fuzzy=0.0, already_used=None) -> list[dict]:
    """
    >>> parse_string_quote_as_spans('de grote ... was lui', 'de grote grijze vos was lui')
    [{'start': 0, 'end': 8, 'text': 'de grote'}, {'start': 20, 'end': 27, 'text': 'was lui'}]
    >>> parse_string_quote_as_spans('de grote ... was lui', 'de grooote grijze vos was lui', fuzzy=.2)
    [{'start': 0, 'end': 8, 'text': 'de grooo'}, {'start': 22, 'end': 29, 'text': 'was lui'}]
    >>> parse_string_quote_as_spans('def', 'abc def ghij abc def ghij', already_used=[])
    [{'start': 4, 'end': 7, 'text': 'def'}]
    >>> parse_string_quote_as_spans('def', 'abc def ghij abc def ghij', already_used=[(4, 7)])
    [{'start': 17, 'end': 20, 'text': 'def'}]
    >>> parse_string_quote_as_spans('And when?', 'What for? And why? And if so, when? And for whom will this be done?')
    Traceback (most recent call last):
    ValueError: No match for And when?
    >>> parse_string_quote_as_spans('And ... when?', 'What for? And why? And if so, when? And for whom will this be done?', fuzzy=0.2)
    Traceback (most recent call last):
    ValueError: Multiple matches for And ... when?
    """

    quote_regex = dotted_quote_to_regex(quote, fuzzy)
    matches = list(quote_regex.finditer(original))

    if not matches:
        raise ValueError(f'No match for {quote}')
    elif len(matches) == 1:
        match = matches[0]
    elif len(matches) > 1:
        if already_used is None:
            raise ValueError(f'Multiple matches for {quote}')
        else:
            for match in matches:
                if match.span(0) not in already_used:
                    already_used.append(match.span(0))
                    break
            else:
                raise ValueError(f'Multiple matches for {quote}')

    spans = []
    for n in range(1, len(match.groups()) + 1):
        start, end = match.span(n)
        spans.append({'start': start, 'end': end, 'text': match.group(n)})

    return spans


def dotted_quote_to_regex(quote: str, fuzzy: float) -> regex.Regex:
    """
    Turn a quote string into a regular expression with optional fuzzy matching.
    Each part of the quote string is put in a regex capturing group.

    >>> dotted_quote_to_regex("The quick brown ... over the ... dog", .2)
    regex.Regex('(?:(The\\ quick\\ brown).+(over\\ the).+(dog)){e<=7}', flags=regex.B | regex.I | regex.V0)
    """
    quote_chunks = quote.split('...')
    clean_quote_chunks = [regex.escape(chunk.strip()) for chunk in quote_chunks]
    # make final question marks optional (because LLM often adds them):
    regex_quote_chunks = [f'({chunk + ("?" if chunk.endswith("?") else "")})' for chunk in clean_quote_chunks]
    the_regex_str = '(?:' + ('.+'.join(regex_quote_chunks)) + ')'

    if fuzzy:
        fuzzy_nchars = int(fuzzy * len(quote))
        the_regex_str += f'{{e<={fuzzy_nchars}}}'

    return regex.compile(the_regex_str, flags=regex.IGNORECASE + regex.BESTMATCH)


if __name__ == '__main__':

    main()