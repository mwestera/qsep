import argparse
import sys
import logging
import json
from transformers import pipeline
import functools


# TODO: Operate on question sequences, not just single questions, given "zoja" etc.

SYSTEM_PROMPT = "You are a system that can break down a potentially complex question and dependent questions into self-contained subquestions, particularly for the Dutch language."
EXAMPLES = [
    {'prompt': 'Sinds wanneer geldt deze maatregel en wat was destijds de motivatie?',
     'response': ['Sinds wanneer geldt deze maatregel?', 'Wat was destijds de motivatie voor deze maatregel?']},
    {'prompt': 'Heeft u de brief van de Indonesische overheid gelezen, en zoja, wat is uw reactie?',
     'response': ["Heeft u de brief van de Indonesische overheid gelezen", "Als u de brief van de Indonesische overheid gelezen heeft, wat is dan uw reactie?"]},
    {'prompt': 'Bent u het met mij eens dat dierenrecht een prominentere plek moet innemen in de samenleving?',
     'response': ['Bent u het met mij eens dat dierenrecht een prominentere plek moet innemen in de samenleving?']},
    {'prompt': 'Wat is de grondwettelijke status van deze maatregel? Is dit onderzocht?',
     'response': ["Wat is de staatrechtelijke grondslag van deze maatregel?", "Is de staatrechtelijke grondslag van deze maatregel onderzocht?"]},
    {'prompt': 'Bent u bekend met het nieuwsbericht dat steeds meer asielzoekers via Luxemburg reizen?',
     'response': ['Bent u bekend met het nieuwsbericht dat steeds meer asielzoekers via Luxemburg reizen?']},
    {'prompt': 'Hoevaak en wanneer nemen mensen in Nederland de fiets? Wat is daarover uw mening?',
     'response': ["Hoevaak nemen mensen in Nederland de fiets?", "Wanneer nemen mensen in Nederland de fiets?", "Wat is uw mening over hoevaak en wanneer mensen in Nederland de fiets nemen?"]},
]


def main():

    logging.basicConfig(level=logging.INFO)

    argparser = argparse.ArgumentParser(description='Qsep')
    argparser.add_argument('file', nargs='?', type=argparse.FileType('r'), default=sys.stdin, help='Input file, one (composite) question per line; when omitted stdin.')
    argparser.add_argument('--model', nargs='?', default="unsloth/llama-3-70b-Instruct-bnb-4bit", type=str)
    # argparser.add_argument('--nudge', nargs='*', type=str)
    argparser.add_argument('--json', action='store_true', help='Whether to give json output; otherwise each question on a new line, with empty line per input.')
    argparser.add_argument('--temp', required=False, type=float, help='Temperature', default=.1)
    argparser.add_argument('--retry', required=False, type=int, help='Max number of retries if response failed to parse.', default=5)
    args = argparser.parse_args()
    # if args.nudge:
    #     args.nudge = ''.join(f'- {nudge}\n' for nudge in args.nudge)

    chat_starts = iter_chat_starts(args.file, EXAMPLES, SYSTEM_PROMPT)
    pipe = functools.partial(pipeline("text-generation", model=args.model), max_new_tokens=1000, temperature=args.temp)
    logging.warning("Feeding transformers.pipeline a list because of transformers inconsistency.")
    for chat_start in chat_starts:
        result = retry_until_parse(pipe, chat_start, parse_json_list_of_strings, args.retry)
        if result is None:
            print()
        else:
            if args.json:
                print(json.dumps(result))
            else:
                for res in result:
                    print(res)
        print()


def retry_until_parse(pipe, chat_start, parser, n_retries):
    n_try = 0
    raw_results = []
    result = None
    while result is None and n_try < n_retries:
        n_try += 1
        raw = pipe([chat_start])[0][0]['generated_text'][-1]['content']
        raw_results.append(raw)
        try:
            result = parser(raw)
        except ValueError:
            continue
        else:
            return result
    else:
        logging.warning(f'Failed parsing: {" / ".join(raw_results)}')
        return None


def parse_json_list_of_strings(raw):
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError
    if not isinstance(result, list):
        raise ValueError
    if any(not isinstance(x, str) for x in result):
        raise ValueError
    return result


def iter_chat_starts(texts, examples, system_prompt):
    examples_chat = []
    for example in examples:
        examples_chat.append({"role": "user", "content": example['prompt']})
        examples_chat.append({"role": "assistant", "content": json.dumps(example['response'])})

    for text in texts:
        text = text.strip()
        chat_start = [
            {"role": "system", "content": system_prompt},
            *examples_chat,
            {"role": "user", "content": text},
        ]
        yield chat_start



if __name__ == '__main__':

    main()