# EntityQuestions
This repository contains the EntityQuestions dataset as well as code to evaluate retrieval results from the the paper [Simple Entity-centric Questions Challenge Dense Retrievers]() by Chris Sciavolino*, Zexuan Zhong*, Jinhyuk Lee, and Danqi Chen (* equal contribution).

## Quick Links
  - [Dataset Overview](#dataset-overview)
  - [Retrieving DPR Results](#retrieving-dpr-results)
  - [Retrieving BM25 Results](#retrieving-bm25-results)
  - [Bugs or Questions?](#bugs-or-questions)
  - [Citation](#citation)

## Dataset Overview
We store all question/answer files using the .jsonl format (if you're unfamiliar, it's essentially a list of JSON objects). If you're interested, we included a file `utils/jsonl_to_json.py` that can convert from .jsonl to .json and vice-versa.

The main dataset is included in `dataset/` under `train/`, `dev/`, and `test/`. We include sampled training datasets, development datasets, and testing datasets in the corresponding directory. For example, the evaluation set for place-of-birth (P19) is located in `dataset/test/P19.test.jsonl`.

We also include all of the one-off datasets we used to generate the tables/figures presented in the paper under `dataset/one-off/`.


## Retrieving DPR Results
Our analysis is based on a previous version of the DPR repository (specifically the Oct. 5 version w. hash [27a8436b070861e2fff481e37244009b48c29c09](https://github.com/facebookresearch/DPR/tree/27a8436b070861e2fff481e37244009b48c29c09)), so our commands may not be up-to-date with the March 2021 release. That said, most of the commands should be clearly transferable.

First, we recommend following the setup guide from the official DPR repository. Once set up, you can download the relevant pre-trained models/indices using their download_data.py script. For our analysis, we used the DPR-NQ model and the DPR-Multi model. To run retrieval using a pre-trained model, you'll minimally need to download:

1. The pre-trained model
2. The Wikipedia passage splits
3. The encoded Wikipedia passage FAISS index
4. A question/answer dataset

With this, you can use the following python command:

``` bash
python dense_retriever.py \
    --batch_size 512 \
    --model_file "path/to/pretrained/model/file.cp" \
    --qa_file "path/to/qa/dataset/to/evaluate.json" \
    --ctx_file "path/to/wikipedia/passage/splits.tsv" \
    --encoded_ctx_file "path/to/encoded/wikipedia/passage/index/" \
    --save_or_load_index \
    --n-docs 100 \
    --validation_workers 1 \
    --out_file "path/to/desired/output/location.json"
```

We had access to a single 11Gb Nvidia RTX 2080Ti GPU w. 128G of RAM when running DPR retrieval. 


## Retrieving BM25 Results
We use the Pyserini implementation of BM25 for our analysis. We use the default settings and index on the same passage splits downloaded from the DPR repository. We include steps to re-create our BM25 results below.

First, we need to pre-process the DPR passage splits into the proper format for BM25 indexing. We include this file in `utils/build_bm25_ctx_passages.py`. Rather than writing all passages into a single file, you can optionally shard the passages into multiple files (specified by the `n_shards` argument). It also creates a mapping from the passage ID to the title of the article the passage is from. You can use this file as follows:

``` bash
python utils/build_bm25_ctx_passages.py \
    --wiki_passages_file "path/to/wikipedia/passage/splits.tsv" \
    --outdir "path/to/desired/output/directory/" \
    --title_index_path "path/to/desired/output/directory/.json" \
    --n_shards number_of_shards_of_passages_to_write
```

Now that you have all the passages in files, you can build the BM25 index using the following command:

``` bash
python -m pyserini.index -collection JsonCollection \
    -generator DefaultLuceneDocumentGenerator \
    -threads 4 \
    -input "path/to/generated/passages/folder/" \
    -index "path/to/desired/index/folder/" \
    -storePositions -storeDocvectors -storeRaw
```


## Bugs or Questions?
Feel free to open an issue on this GitHub repository and we'd be happy to answer your questions as best we can!

## Citation
If you use our dataset or code in your research, please cite our work:
```bibtex
@inproceedings{sciavolino2021simple,
   title={Simple Entity-centric Questions Challenge Dense Retrievers},
   author={Sciavolino, Chris and Zhong, Zexuan and Lee, Jinhyuk and Chen, Danqi},
   booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
   year={2021}
}
```