# QSep: Question separator #

A command-line tool for breaking down potentially composite questions into their subquestions, using an LLM.

Currently intended for Dutch, but easily adapted.

## Install ##

Ideally in a virtual environment (or use `pipx`):

```bash
pip install git+https://github.com/mwestera/qsep
```

## Usage ##

Given a text file `questions.txt` containing questions to break down, one per line:

```text
Hoe werkt dat en waarom?
Wie ben je en hoe oud ben je?
```

You can feed it into `qsep` like this (for example; default temperature is 0.1):

```bash
cat questions.txt | qsep --temp .3
```

This will output one subquestion per line, the outputs for different inputs separated by empty lines.
Alternatively, add `--json` to get JSON lists as output, one per input.