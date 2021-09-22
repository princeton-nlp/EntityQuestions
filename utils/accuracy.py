import argparse
import glob
import json
from collections import defaultdict
from tqdm import tqdm

import utils.ion as ion
from utils.has_answer_fn import HAS_ANS_FNS
from utils.tokenizers import SimpleTokenizer


AVERAGE = "Average"
DEFAULT_TOKENIZER = SimpleTokenizer()


def topk_retrieval_accuracy(results_lst, k_values, has_answer_fn, tokenizer=DEFAULT_TOKENIZER):
    """
    Input:
    - results_lst: List[Dict]. Output from a retriever with retrieved contexts for each question.
    - k_values: List[int]. List of number of contexts to consider for retrieval accuracy.
    - has_answer_fn: Function(ctx, answers) -> bool. Function that returns true iff the context
        dictionary contains at least one answer.
    - tokenizer: Tokenizer=SimpleTokenizer. Takes text and parses it into a list of word tokens.

    Output:
    - k_accuracy: Dict[int, float]. Map from k-value to retrieval accuracy, represented as a float.
    """
    k_num_correct = defaultdict(int)
    for result in tqdm(results_lst):
        answer_lst = result['answers']
        ctxs = result['ctxs']
        found_k = -1
        for ctx_idx, ctx in enumerate(ctxs):
            if has_answer_fn(ctx, answer_lst, tokenizer):
                found_k = ctx_idx
                break
        for k in k_values:
            if found_k > -1 and found_k < k:
                k_num_correct[k] += 1

    num_examples = len(results_lst)
    k_accuracy = { k: n_correct / num_examples for k,n_correct in k_num_correct.items() }
    return k_accuracy


def main():
    args = parse_args()
    if args.glob:
        results_files = glob.glob(args.results)
    else:
        results_files = [args.results]

    has_answer_fn = HAS_ANS_FNS[args.answer_match]
    k_values = sorted([int(k) for k in args.k_values.split(',')])
    file_k_accuracy = defaultdict(dict)
    for file in results_files:
        results_lst = ion.read_json(file, log=True)
        file_k_accuracy[file] = topk_retrieval_accuracy(results_lst, k_values, has_answer_fn)
    
    if len(results_files) > 1:
        k_avg_accuracy = defaultdict(int)
        for file in results_files:
            for k in k_values:
                k_avg_accuracy[k] += file_k_accuracy[file][k]
        k_avg_accuracy = { k: acc_sum / len(results_files) for k, acc_sum in k_avg_accuracy.items() }
        file_k_accuracy[AVERAGE] = k_avg_accuracy
    
    file_pretty_name = {file: file.split('/')[-1] for file in results_files}
    file_pretty_name[AVERAGE] = AVERAGE
    results_files_sorted = sorted(results_files, key=lambda f: file_pretty_name[f])
    results_files_sorted = results_files_sorted + [AVERAGE] if len(results_files) > 1 else results_files_sorted

    table_lines = []
    for file in results_files_sorted:
        table_line = []
        for k in k_values:
            table_line.append('{:.1%}'.format(file_k_accuracy[file][k]))
        table_line.append(file_pretty_name[file])
        table_lines.append(table_line)
    
    header = ["k={}".format(k) for k in k_values] + ["filename"]
    print('\t'.join(header))
    for line in table_lines:
        print('\t'.join(line))
    

def check_args(args):
    k_values = sorted([int(k) for k in args.k_values.split(',')])
    print(f'k_values = [ {", ".join([str(k) for k in k_values])} ]')
    assert args.answer_match in HAS_ANS_FNS, f'Invalid args.answer_match. Must be in: [{", ".join(HAS_ANS_FNS.keys())}]'


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--results', required=True, type=str, help='Filepath (or glob if --glob is set) to retrieval results to evaluate.')
    parser.add_argument('--glob', action='store_true', help='True if --results denotes a glob of files.')
    parser.add_argument('--k_values', type=str, help='Comma separated values of K to find retrieval accuracy for. e.g. 1,5,20,100')
    parser.add_argument('--answer_match', type=str, default='has_answer', help='Type of function to use to determine answer presence for' +
                        'a context. Supported: { has_answer, string, regex, title }. Default: has_answer.')

    args = parser.parse_args()
    check_args(args)
    return args


if __name__ == '__main__':
    main()
