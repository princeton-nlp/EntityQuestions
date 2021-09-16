"""
General framework for reading/writing data from/to files on
the local system.
"""

import json
import random
from pathlib import Path


###############################################################################
#   FILE READING              #################################################
###############################################################################

def read_file(infile, handle_file, log=False, skip_first_line=False):
    if log:
        print('Opening "{}"...'.format(infile))
    data = None
    with open(infile) as f:
        if skip_first_line:
            f.readline()
        data = handle_file(f)
    if log:
        print('  Done.')
    return data


def read_json(infile, log=False):
    handler = lambda f: json.load(f)
    return read_file(infile, handler, log=log)


def read_jsonl(infile, log=False):
    handler = lambda f: [json.loads(line) for line in f.readlines()]
    return read_file(infile, handler, log=log)


def read_tsv(infile, row_fn=lambda x: x, log=False, skip_first_line=False):
    handler = lambda f: [row_fn(line.split('\t')) for line in f.readlines()]
    return read_file(infile, handler, log=log, skip_first_line=skip_first_line)


###############################################################################
#   FILE WRITING              #################################################
###############################################################################

def write_file(outfile, handle_file, log=False):
    if log:
        print('Writing to "{}"...'.format(outfile))
    with open(outfile, 'w+') as f:
        handle_file(f)
    if log:
        print('  Done.')


def write_json(outfile, data, log=False, pretty=False):
    handler = lambda f: f.write(json.dumps(data, indent=4 if pretty else None))
    write_file(outfile, handler, log=log)


def write_jsonl(outfile, data, log=False):
    def _write_jsonl(f):
        for dct in data:
            f.write(json.dumps(dct) + '\n')
    handler = _write_jsonl
    write_file(outfile, handler, log=log)


def mkdir_optional(outdir):
    Path(outdir).mkdir(parents=True, exist_ok=True)


###############################################################################
#   OTHER OUTPUT              #################################################
###############################################################################
def sample_print(item, pct):
    if random.random() < (pct / 100):
        print(item)
