# EntityQuestions
This repository contains the EntityQuestions dataset as well as code to evaluate retrieval results from the the paper [Simple Entity-centric Questions Challenge Dense Retrievers](https://github.com/princeton-nlp/EntityQuestions/blob/master/paper.pdf) by Chris Sciavolino*, Zexuan Zhong*, Jinhyuk Lee, and Danqi Chen (* equal contribution).

*[9/16/21] This repo is not yet set in stone, we're still putting finishing touches on the tooling and documentation :)*

## Quick Links
  - [Dataset Overview](#dataset-overview)
  - [Retrieving DPR Results](#retrieving-dpr-results)
  - [Retrieving BM25 Results](#retrieving-bm25-results)
  - [Evaluating Retriever Results](#evaluating-retriever-results)
  - [Bugs or Questions?](#bugs-or-questions)
  - [Citation](#citation)

## Dataset Overview
You can download a `.zip` file of the dataset [here](https://nlp.cs.princeton.edu/projects/entity-questions/dataset.zip), or using `wget` with the command:

``` bash
wget https://nlp.cs.princeton.edu/projects/entity-questions/dataset.zip
```

The unzipped directory should have the following structure:

```
dataset/
    | train/
        | P*.train.json     // all randomly sampled training files 
    | dev/
        | P*.dev.json       // all randomly sampled development files
    | test/
        | P*.test.json      // all randomly sampled testing files
    | one-off/
        | common-random-buckets/
            | P*/
                | bucket*.test.json
        | no-overlap/
            | P*/
                | P*_no_overlap.{train,dev,test}.json
        | nq-seen-buckets/
            | P*/
                bucket*.test.json
        | similar/
            | P*
                | P*_similar.{train,dev,test}.json
```

The main dataset is included in `dataset/` under `train/`, `dev/`, and `test/`, each containing the randomly sampled training, development, and testing subsets, respectively. For example, the evaluation set for place-of-birth (P19) can be found in the `dataset/test/P19.test.json` file.

We also include all of the one-off datasets we used to generate the tables/figures presented in the paper under `dataset/one-off/`, explained below:

- `one-off/common-random-buckets/` contains buckets of 1,000 randomly sampled examples, used to produce Fig. 1 of the paper (specifically for `rand-ent`).
- `one-off/no-overlap/` contains the training/development splits for our analyses in Section 4.1 of the paper (we do not use the testing split in our analysis). These training/development sets have subject entities with no token overlap with subject entities of the randomly sampled test set (specifically for all fine-tuning in Table 2).
- `one-off/nq-seen-buckets/` contains buckets of questions with subject entities that overlap with subject entities seen in the NQ training set, used to produce Fig. 1 of the paper (specifically for `train-ent`).
- `one-off/similar` contains the training/development splits for the syntactically different but symantically equal question sets, used for our analyses in Section 4.1 (specifically the `similar` rows). Again, we do not use the testing split in our analysis. These questions are identical to `one-off/no-overlap/` but use a different question template.


## Retrieving DPR Results
Our analysis is based on a previous version of the DPR repository (specifically the Oct. 5 version w. hash [27a8436b070861e2fff481e37244009b48c29c09](https://github.com/facebookresearch/DPR/tree/27a8436b070861e2fff481e37244009b48c29c09)), so our commands may not be up-to-date with the March 2021 release. That said, most of the commands should be clearly transferable.

First, we recommend following the setup guide from the official DPR repository. Once set up, you can download the relevant pre-trained models/indices using their `download_data.py` script. For our analysis, we used the DPR-NQ model and the DPR-Multi model. To run retrieval using a pre-trained model, you'll minimally need to download:

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
We use the [Pyserini](https://github.com/castorini/pyserini/) implementation of BM25 for our analysis. We use the default settings and index on the same passage splits downloaded from the DPR repository. We include steps to re-create our BM25 results below.

First, we need to pre-process the DPR passage splits into the proper format for BM25 indexing. We include this file in `bm25/build_bm25_ctx_passages.py`. Rather than writing all passages into a single file, you can optionally shard the passages into multiple files (specified by the `n_shards` argument). It also creates a mapping from the passage ID to the title of the article the passage is from. You can use this file as follows:

``` bash
python bm25/build_bm25_ctx_passages.py \
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

Once the index is built, you can use it in the `bm25/bm25_retriever.py` script to get retrieval results for an input file:

``` bash
python bm25/bm25_retriever.py \
    --index_path "path/to/built/bm25/index/directory/" \
    --passage_id_to_title_path "path/to/title_index_path/from_step_1.json" \
    --input "path/to/input/qa/file.json" \
    --output_dir "path/to/output/directory/"
```

By default, the script will retrieve 100 passages (`--n_docs`), use string matching to determine answer presence (`--answer_type`), and take in `.json` files (`--input_file_type`). You can optionally provide a glob using the `--glob` flag. The script writes the results to the file with the same name as the input file, but in the output directory.


## Evaluating Retriever Results
We provide an evaluation script in `utils/accuracy.py`. The expected format is equivalent to DPR's output format. It either accepts a single file to evaluate, or a glob of multiple files if the `--glob` option is set. To evaluate a single file, you can use the following command:

``` bash
python utils/accuracy.py \
    --results "path/to/retrieval/results.json" \
    --k_values 1,5,20,100
```

or with a glob with:

``` bash
python utils/accuracy.py \
    --results="path/to/glob*.test.json" \
    --glob \
    --k_values 1,5,20,100
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
