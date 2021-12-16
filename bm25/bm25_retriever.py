import argparse
import glob
import json

from pyserini.search import SimpleSearcher
from tqdm import tqdm

import utils.ion as ion
import utils.tokenizers as tokenizers

from utils.has_answer_fn import HAS_ANS_FNS


OPEN_FNS = { 'json': ion.read_json, 'jsonl': ion.read_jsonl }


def search(dataset, n_docs, has_answer_fn, searcher, pid2title):
    """
    Input
    - dataset: List[Dict]. List of question/answer entries
    - n_docs: int. Number of contexts to retrieve per question.
    - has_answer_fn: Function(context, answers) -> bool. Function to determine
        if a context contains one of the answers.
    - searcher: pyserini.search.SimpleSearcher. Pyserini object in charge of search
    - pid2title: Dict[str, str]. Mapping of PassageId -> Title

    Output
    - results: List[Dict]. List of retrieval results for each question/answer pair in
        the dataset.
    """
    results = []
    for entry in tqdm(dataset):
        question = entry['question']
        answer_lst = entry['answers']
        hits = searcher.search(question, k=n_docs)
        titles = [pid2title[hit.docid] for hit in hits]
        ctxs = [{
            'id': hit.docid,
            'title': title,
            'text': json.loads(hit.raw)['contents'][len(title):].strip(),
            'score': hit.score,
        } for hit, title in zip(hits, titles)]
        has_answers = [has_answer_fn(ctx, answer_lst) for ctx in ctxs]
        ctxs = [{ **ctx, 'has_answer': has_answer } for ctx, has_answer in zip(ctxs, has_answers)]
        results.append({ 'question': question, 'answers': answer_lst, 'ctxs': ctxs })
    return results


def main():
    args = parse_args()

    qa_files = glob.glob(args.input) if args.glob else [args.input]
    print('Input files: [ {} ]'.format(', '.join(qa_files)))

    has_answer_fn = HAS_ANS_FNS[args.answer_type]
    open_fn = OPEN_FNS[args.input_file_type]
    pid2title = ion.read_json(args.passage_id_to_title_path, log=True)
    searcher = SimpleSearcher(args.index_path)

    for qa_file in qa_files:
        dataset = open_fn(qa_file, log=True)
        results = search(dataset, args.n_docs, has_answer_fn, searcher, pid2title)
        outfile = args.output_dir + qa_file.split('/')[-1]
        ion.write_json(outfile, results, log=True, pretty=True)


def check_args(args):
    assert args.answer_type in HAS_ANS_FNS, "Supported args.answer_type: [{}]".format(','.join(HAS_ANS_FNS.keys()))
    assert args.input_file_type in OPEN_FNS, 'Supported args.input_file_type: [{}]'.format(','.join(OPEN_FNS.keys()))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_path', type=str, required=True, help='Location of preprocessed BM25 index.')
    parser.add_argument('--passage_id_to_title_path', type=str, required=True, help='Location of mapping from passage ID to titile.')
    parser.add_argument("--input", type=str, required=True, help="Run BM25 retrieval on this input.")
    parser.add_argument('--output_dir', type=str, required=True, help='Location of output directory where results will be dumped.')

    # optional arguments
    parser.add_argument('--glob', action='store_true', help='Set this flag if --input is a glob.')
    parser.add_argument('--n_docs', type=int, default=100, help='Number of contexts to retrieve per question.')
    parser.add_argument('--answer_type', type=str, default='string', help='How to determine answer presence.')
    parser.add_argument('--input_file_type', type=str, default='json', help='Format of the input questions.')

    args = parser.parse_args()
    check_args(args)
    return args

if __name__ == '__main__':
    main()
