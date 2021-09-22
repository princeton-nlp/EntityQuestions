"""
Build passage shards
"""
import argparse
import os
import utils.ion as ion
from tqdm import tqdm


def read_passages(wikipath):
    """
    Input
    - wikipath: str. Filepath to a list of all Wikipedia passage splits.

    Output
    - passages: List[Dict]. List of all processed Wikipedia passage splits.
    - pid2title: Dict[str, str]. Mapping of PassageId -> Title.
    """
    pid2title = {}
    def _serialize(split):
        pid = split[0].strip().lower()
        title = split[2].strip()
        pid2title[pid] = title
        text = split[1].strip()[1:-1]   # Omits extra quotations
        return { 'id': pid, 'contents': f'{title} {text}' }
    passages = ion.read_tsv(wikipath, row_fn=_serialize, log=True)
    return passages, pid2title


def prepare_passages_to_index(wikipath, outdir, title_index_path, n_shards=10):
    """
    Input
    - wikipath: str. Filepath to a list of all Wikipedia passage splits.
    - outdir: str. Filepath to output folder where to dump processed passage shards.
    - title_index_path: str. Filepath where to dump PassageId -> Title mapping.
    - n_shards: int. Number of files to divide passages over.

    Output
    - Write all passage shards to [outdir] and mapping to [title_index_path].
    
    """
    outpath_template = outdir + 'shard{}.json'
    passages, pid2title = read_passages(wikipath)
    shard_size = int(len(passages) / n_shards) + 1
    for shard_id in tqdm(range(n_shards)):
        shard = passages[shard_id*shard_size:(shard_id+1)*shard_size]
        outpath = outpath_template.format(shard_id)
        ion.write_json(outpath, shard, log=True)
    ion.write_json(title_index_path, pid2title, log=True)
    print('Total number of passages: {}'.format(len(passages)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--wiki_passages_file', required=True, type=str, default=None,
                        help="Location of the Wikipedia passage splits.")
    parser.add_argument('--outdir', required=True, type=str, default=None,
                        help="Directory where the passage shards will be dumped.")
    parser.add_argument('--title_index_path', type=str, help="Location where to write "+ 
                        "index of passage_id --> title.")
    parser.add_argument('--n_shards', type=int, default=10,
                        help="Number of shards to split the passages into.")
    
    args = parser.parse_args()
    assert args.wiki_passages_file is not None
    assert args.outdir is not None
    prepare_passages_to_index(args.wiki_passages_file, args.outdir, args.title_index_path, args.n_shards)
