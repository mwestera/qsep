import argparse
import sys
import logging
import json
from transformers import pipeline


# TODO: Operate on question sequences, not just single questions, given "zoja" etc.

SYSTEM_PROMPT = "You are a system that can reformulate questions to isolate the subquestions they contain, particularly for the Dutch language."
EXAMPLES = [
    {'prompt': 'Sinds wanneer geldt deze maatregel en wat was destijds de motivatie?',
     'response': ['Sinds wanneer geldt deze maatregel?', 'Wat was destijds de motivatie voor deze maatregel?']},
    {'prompt': 'Heeft u de brief van de Indonesische overheid gelezen, en zoja, wat is uw reactie?',
     'response': ["Heeft u de brief van de Indonesische overheid gelezen", "Wat is uw reactie op de brief van de Indonesische overheid?"]},
    {'prompt': 'Bent u het met mij eens dat dierenrecht een prominentere plek moet innemen in de samenleving?',
     'response': ['Bent u het met mij eens dat dierenrecht een prominentere plek moet innemen in de samenleving?']},
    {'prompt': 'Wat is de grondwettelijke status van deze maatregel? Is dit onderzocht?',
     'response': ["Wat is de staatrechtelijke grondslag van deze maatregel?", "Is de staatrechtelijke grondslag van deze maatregel onderzocht?"]},
    {'prompt': 'Bent u bekend met het nieuwsbericht dat steeds meer asielzoekers via Luxemburg reizen?',
     'response': ['Bent u bekend met het nieuwsbericht dat steeds meer asielzoekers via Luxemburg reizen?']},
    {'prompt': 'Hoe vaak en wanneer nemen mensen in Nederland de fiets?',
     'response': ["Hoe vaak nemen mensen in Nederland de fiets?", "Wanneer nemen mensen in Nederland de fiets?"]},
]

def main():

    logging.basicConfig(level=logging.INFO)

    argparser = argparse.ArgumentParser(description='Qsep')
    argparser.add_argument('--model', nargs='?', default="unsloth/llama-3-70b-Instruct-bnb-4bit", type=str)
    # argparser.add_argument('--nudge', nargs='*', type=str)
    argparser.add_argument('--json', action='store_true', help='Whether to give json output; otherwise each question on a new line, with empty line per input.')
    args = argparser.parse_args()
    # if args.nudge:
    #     args.nudge = ''.join(f'- {nudge}\n' for nudge in args.nudge)

    chat_starts = iter_chat_starts(sys.stdin, EXAMPLES, SYSTEM_PROMPT)
    pipe = pipeline("text-generation", model=args.model)
    logging.warning("Feeding transformers.pipeline a list because of transformers inconsistency.")
    for responses in pipe(list(chat_starts), max_new_tokens=1000):  # TODO Fix once fixed
        for response in responses:
            raw = response['generated_text'][-1]['content']
            try:
                result = parse_response(raw)
            except ValueError:
                logging.warning(f'Failed parsing: {raw}')
                print()
                continue

            if args.json:
                print(json.dumps(result))
            else:
                for res in result:
                    print(res)
        print()


def parse_response(raw):
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