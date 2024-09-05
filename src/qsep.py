import argparse
import sys
import logging
import functools

from transformers import pipeline


# TODO: Operate on question sequences, not just single questions, given "zoja" etc.

SYSTEM_PROMPT = "Je bent een behulpzame assistent, en een taalkundig expert op het gebied van vragen."
PROMPT = """Soms worden in 1 zin meerdere vragen gesteld. Voorbeelden: 

Voorbeeld 1: Hoe heten Ben en Piet?
Voorbeeld 2: Hoe laat is het en waar ga je naartoe?
Voorbeeld 3: Heb je die brief gelezen en zoja, wat vond je ervan? 
Voorbeeld 4: Waarom en sinds wanneer werkt dat zo?

Kun je me helpen? Iemand stuurde me deze vraag:

> {text}

Worden hier meerdere vragen gesteld, of maar eentje?

Schrijf alle afzonderlijke vragen in een lijst (met `-` als bullets). Bedankt!
"""

def main():

    logging.basicConfig(level=logging.INFO)

    argparser = argparse.ArgumentParser(description='Qsep')
    argparser.add_argument('--model', nargs='?', default="unsloth/llama-3-70b-Instruct-bnb-4bit", type=str)
    argparser.add_argument('--nudge', nargs='*', type=str)
    args = argparser.parse_args()
    if args.nudge:
        args.nudge = ''.join(f'- {nudge}\n' for nudge in args.nudge)

    chat_starts = iter_chat_starts(sys.stdin, SYSTEM_PROMPT, functools.partial(PROMPT.format, nudge=args.nudge))
    pipe = pipeline("text-generation", model=args.model)
    logging.warning("Feeding transformers.pipeline a list because of transformers inconsistency.")
    for responses in pipe(list(chat_starts), max_new_tokens=1000):  # TODO Fix once fixed
        for response in responses:
            print(response['generated_text'][-1]['content'])



def iter_chat_starts(texts, system_prompt, prompt_format):
    for text in texts:
        text = text.strip()
        chat_start = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_format(text=text)},
        ]
        yield chat_start


if __name__ == '__main__':

    main()