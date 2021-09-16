import argparse
import glob
import json
import utils.ion as ion


def main():
    args = parse_args()

    input_files = glob.glob(args.input_glob)
    for file in input_files:
        outfile = args.output_dir + '.'.join(file.split('/')[-1].split('.')[:-1])
        if args.json_to_jsonl:
            outfile = outfile + '.jsonl'
            dataset = ion.read_json(file)
            ion.write_jsonl(outfile, dataset, log=args.verbose)
        else:
            outfile = outfile + '.json'
            dataset = ion.read_jsonl(file)
            ion.write_json(outfile, dataset, log=args.verbose, pretty=args.pretty_json)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_glob', required=True, type=str, help='Glob of all input files to re-write.')
    parser.add_argument('--output_dir', required=True, type=str, help='Location of where to dump re-written results')
    parser.add_argument('--json_to_jsonl', action='store_true', help='Convert from .json to .jsonl')
    parser.add_argument('--verbose', action='store_true', help='Print out input/output file information.')
    parser.add_argument('--pretty_json', action='store_true', help='Write .json file with indent.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()