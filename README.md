# Mitigating Label mismatches across datasets with Meta-Learning
Research project: Mitigating Label Mismatches across Datasets using Meta-Learning (document classification)

## Installation
The environment can be created with 

```conda env create -f atcs2.yml```

If there is difficulty with installing the huggingface datasets, do it manually with 

```conda install -c huggingface -c conda-forge datasets```

## Code hygiene
Please keep a separate branch for developing each of the three tasks. Once your functionality is completely tested and ready (not v0.3, but v1.0), merge it into the main branch. If you want to add a new functionality to a branch that requires (major) changes to existing code, create a new branch, finish your thing, merge it back.
I recommend github desktop for convenience.
If you want to be really proper: If you want to add **any** new functionality to a branch, create a new branch and follow the same procedure.

For datasets, please maintain/create a structure of 'Data/DatasetName/TrainOrTestOrDevSplit/files'. Everything in the Data folder is ignored, so please add an automatic download function in your code, and/or provide download instructions in this README.

If your functionalities are done, add them to main and make them callable from the command line.

Furthermore, maintain general hygiene:
- Define classes (e,g. lightning models), functionalities (e.g. cleaning the dataset), and experiments (e.g. finetune bert) in separate files.
- Update atcs2.yml file if you install new libraries
- Separate functions as much as possible
- Comment your code!!1!
- stick to the styleguide https://pep8.org/

## Datasets
HuffPost News Category Dataset ('hp'): This dataset contains 200k documents, each with a headline and a short description. The documents are annotated with a fine grained label of its topic. Due to the size of the dataset, it might be worth to exclude a few labels during training, and use those documents for few-shot evaluation later. 
Link: https://huggingface.co/datasets/Fraser/news-category-dataset 


AG Corpus of News Articles ('ag'): This dataset contains documents with 4 categories. Check with the related work how you commonly preprocess it. 
Related Work: https://paperswithcode.com/sota/text-classification-on-ag-news, http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html 
Link: https://huggingface.co/datasets/ag_news 


20 Newsgroup ('ng'): A dataset with 20 categories that are quite different from the previous datasets. Some categories are very closely related while others are rather coarse. This dataset might require some finetuning in terms of which labels to use, and which to merge. 
Link: https://huggingface.co/datasets/newsgroup 


BBC News classification ('bbc'): A small dataset on classification of BBC news articles. 
Link: https://www.kaggle.com/c/learn-ai-bbc 


DBpedia14 ('dbpedia' when running multitask or protomaml, else 'db'): This dataset contains documents with 14 non-overlapping categories/ontologies the DBpedia, a multi-lingual knowledge base. Almost all documents are in English. Thsi dataset contains 630k documents.
Link: https://huggingface.co/datasets/dbpedia_14

## BERT
The [BERT-base model from Hugging Face](https://huggingface.co/bert-base-uncased) can finetuned with a single fully connected layer on top on a given dataset. The performance of this model will serve as a baseline for the Meta-Learning and Multitask Learning models.
### Usage
The `train_bert.py` can be called as follows:

``python train_bert.py [-h] [--path PATH] [--optimizer {Adam,SGD}] [--lr LR] [--max_epochs MAX_EPOCHS] [--finetuned_layers FINETUNED_LAYERS] [--tokenizer {BERT}] [--batch-size BATCH_SIZE] [--device {cpu,gpu}] [--seed SEED]
                     [--progress_bar] [--max_text_length MAX_TEXT_LENGTH]
                     name {hp,ag,bbc} nr_classes``

By default the model will use the Adam optimizer to learn only the fully connected layer for 100 epochs with a constant learning rate of `0.001`. Also, by default there is no limit on the text lenght, but the BERT model does not work with text longer than `512` characters.

For more information run `python train_bert.py -h`.

## Multitask model
The Multitask model can be finetuned with mutliple train sets. Provide the train sets that you want to train as a string of (abbreviated) dataset names in the argument `--train_sets`. The `train_multitask.py` can be called as follows:

``python train_multitask.py [-h] [--path PATH] [--optimizer {Adam,SGD}]
                          [--lr LR] [--max_epochs MAX_EPOCHS]
                          [--finetuned_layers FINETUNED_LAYERS]
                          [--task_layers TASK_LAYERS] [--tokenizer {BERT}]
                          [--batch_size BATCH_SIZE] [--device {cpu,cuda}]
                          [--seed SEED] [--progress_bar]
                          [--max_text_length MAX_TEXT_LENGTH]
                          [--sample SAMPLE] [--num_workers NUM_WORKERS]
                          [--gpus GPUS] [--train_sets TRAIN_SETS]
                          [--test_set TEST_SET] [--hidden HIDDEN]
                          [--max_tokens MAX_TOKENS]
                          [--inner_steps INNER_STEPS] [--inner_lr INNER_LR]
                          [--shot SHOT] [--query_batch QUERY_BATCH]
                          [--eval_perm EVAL_PERM] [--eval_way EVAL_WAY]
                          [--class_eval CLASS_EVAL] [--data_path DATA_PATH]
                          [--cache_path CACHE_PATH]
                          name``

By default, the model will be trained on the HuffPost News Category, DBpedia14 and the 20NewsGroup datasets (i.e. `--train_sets=hp,dbpedia,ng`). 
For more information run `python train_multitask.py -h`
                          
To perform few-shot evaluation with the model, specify only one dataset in the argument `--test_set` (e.g. 'bbc' for the BBC dataset) Then run `evaluate_mutlitask.py` which can ce balled as follows:

``python evaluate_multitask.py [-h] [--path PATH] [--optimizer {Adam,SGD}]
                             [--lr LR] [--max_epochs MAX_EPOCHS]
                             [--finetuned_layers FINETUNED_LAYERS]
                             [--task_layers TASK_LAYERS] [--tokenizer {BERT}]
                             [--batch_size BATCH_SIZE] [--device {cpu,cuda}]
                             [--seed SEED] [--progress_bar]
                             [--max_text_length MAX_TEXT_LENGTH]
                             [--sample SAMPLE] [--num_workers NUM_WORKERS]
                             [--gpus GPUS] [--train_sets TRAIN_SETS]
                             [--test_set TEST_SET] [--hidden HIDDEN]
                             [--max_tokens MAX_TOKENS]
                             [--inner_steps INNER_STEPS] [--inner_lr INNER_LR]
                             [--shot SHOT] [--query_batch QUERY_BATCH]
                             [--eval_perm EVAL_PERM] [--eval_way EVAL_WAY]
                             [--class_eval CLASS_EVAL] [--data_path DATA_PATH]
                             [--cache_path CACHE_PATH]
                             name``

Make sure to fill in `name` the name of the checkpoint path of the mutlitask model without the extension `.ckpt`. The checkpoint path of the multitask model can be downloaded here: https://drive.google.com/file/d/1pjCplwjC4f4yMZMEXGsaoyTcxCkxzPt0/view?usp=sharing

To calculate the mean accuracies for the multitask model run the script `test_multitask.sh` in your terminal.

## ProtoMAML
To run the ProtoMAML algorithm, run `protomaml.py` which can be called as follows:

```python protomaml.py```

ProtoMAML has been trained with the same sets of train sets as the multitask model (i.e. hp, dbpedia and ng) and evaluated on the same test sets (i.e. ag and bbc)
