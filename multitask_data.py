import torch
from torch.utils import data
from transformers import BertTokenizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import re
import os
import json


tokenizers = {
    "BERT" : BertTokenizer.from_pretrained('bert-base-uncased',  model_max_length=512)
}

class LoadMultitaskData():
    def __init__(self, conf):
        self.load_datasets(conf)

    def load_hp(self, conf):
        """ Loads the HuffPost (hp) dataset from [Hugging Face](https://huggingface.co/datasets/Fraser/news-category-dataset)
            Splits in train (160682), validation (10043) and test (30128). Every sample has the following features:
            `['authors', 'category', 'category_num', 'date', 'headline', 'link', 'short_description']` """

        dataset = load_dataset("Fraser/news-category-dataset")
        train = self.process_hp(dataset["train"][:conf.sample], tokenizers[conf.tokenizer], conf.max_text_length)
        val = self.process_hp(dataset["validation"][:conf.sample], tokenizers[conf.tokenizer], conf.max_text_length)
        test = self.process_hp(dataset["test"][:conf.sample], tokenizers[conf.tokenizer], conf.max_text_length)
        return train, val, test


    def process_hp(self, data, tokenizer, max_text_length=-1):
        """ Extracts the headlines and labels from the HuffPost dataset.
            The description is empty for some samples, which is why it is not returned."""
        txt = [head + ' ' + desc for head, desc in zip(data["headline"], data["short_description"])]
        maxlength = [ex[:max_text_length] for ex in txt]
        txt = tokenizer(maxlength, padding=True, return_tensors='pt', truncation=True, max_length=512)['input_ids']
        labels = torch.LongTensor(data["category_num"])
        return [{"txt" : h, "label" : l} for h, l in zip(txt, labels)]


    def load_ag(self, conf):
        """ Loads the AG news dataset from [Hugging Face](https://huggingface.co/datasets/ag_news)
            Splits in train (120000) and test (7600). Every sample has the following features: `['label', 'text']` """

        dataset = load_dataset("ag_news")
        sample = conf.sample * 5 if conf.sample is not None else None
        train = self.process_ag(dataset["train"][:sample], tokenizers[conf.tokenizer], conf.max_text_length)
        train, val = train_test_split(train, test_size=0.2, random_state=42)
        train = train[:conf.sample]
        test = self.process_ag(dataset["test"][:conf.sample], tokenizers[conf.tokenizer], conf.max_text_length)
        return train, val, test


    def process_ag(self, data, tokenizer, max_text_length=-1):
        """ Extracts the headlines and labels from the AG news dataset.
            The headlines contain '\\' characters in place of newlines. """
        maxlength = [ex[:max_text_length] for ex in data["text"]]
        noslash = [ex.replace("\\", " ") for ex in maxlength]
        headlines = tokenizer(noslash, padding=True, return_tensors='pt', truncation=True, max_length=512)['input_ids']
        labels = torch.LongTensor(data["label"])
        return [{"txt" : h, "label" : l} for h, l in zip(headlines, labels)]


    def load_bbc(self, conf):
        """ Loads the BBC news dataset from [Kaggle](https://www.kaggle.com/c/learn-ai-bbc)
            This dataset has to be downloaded manually using the `downloadbbcdata` file.
            Splits in train (1490) and test (735)."""
        l2i = {'entertainment':0, 'business':1, 'politics':2, 'sport':3, 'tech':4}
        # first row are colum names: ['ArticleId' 'Text' 'Category']
        print('\nLoading BBC Dataset\n')
        train_sample = (conf.sample * 5 + 1) if conf.sample is not None else None
        with open('Data/BBC/Train/BBC News Train.csv') as train_file:
            train = [line.replace("\n", "").split(",") for line in train_file][1:train_sample]
            train = [[ex[1], l2i[ex[2]]] for ex in train]

        # first row are colum names: ['ArticleId' 'Text']
        test_sample = conf.sample + 1 if conf.sample is not None else None
        with open(r'Data/BBC/Test/BBC News Test.csv') as test_file:
            test_ex = [line.replace("\n", "").split(",") for line in test_file][1:test_sample]

        # first row are colum names: ['ArticleId' 'Category']
        with open(r'Data/BBC/Test/BBC News Sample Solution.csv') as test_labels_file:
            test_labels = [line.replace("\n", "").split(",") for line in test_labels_file][1:test_sample]

        test = [[ex[1], l2i[label[1]]] for ex, label in zip(test_ex, test_labels) if ex[0] == label[0]]

        train = self.process_bbc(train, tokenizers[conf.tokenizer], conf.max_text_length)
        train, val = train_test_split(train, test_size=0.2, random_state=42)
        train = train[:conf.sample]
        test = self.process_bbc(test, tokenizers[conf.tokenizer], conf.max_text_length)

        assert len(test) == len(test_labels)
        "The BBC test datafiles were corrupted!"

        return train, val, test


    def process_bbc(self, data, tokenizer, max_text_length=-1):
        """ Extracts the headlines and labels from the BBC news dataset. """
        maxlength = [b[0][:max_text_length] for b in data]
        headlines = tokenizer(maxlength, padding=True, return_tensors='pt', truncation=True, max_length=512)['input_ids']

        labels = torch.LongTensor([b[1] for b in data])

        return [{"txt" : h, "label" : l} for h, l in zip(headlines, labels)]


    # See newsgroup.txt for info
    def load_ng(self, conf):
        twenty = False # placeholder for conf.twenty
        text = []

        if twenty:
            categories = ['18828_alt.atheism', '18828_comp.graphics', '18828_comp.os.ms-windows.misc', '18828_comp.sys.ibm.pc.hardware',
            '18828_comp.sys.mac.hardware', '18828_comp.windows.x', '18828_misc.forsale', '18828_rec.autos', '18828_rec.motorcycles',
            '18828_rec.sport.baseball', '18828_rec.sport.hockey', '18828_sci.crypt', '18828_sci.electronics', '18828_sci.med', '18828_sci.space',
            '18828_soc.religion.christian', '18828_talk.politics.guns', '18828_talk.politics.mideast', '18828_talk.politics.misc',
            '18828_talk.religion.misc']

            for i, category in enumerate(categories):
                data = load_dataset("newsgroup", category)['train']['text']
                text += [[" ".join(ex.split('\n', 2)[2].split()[:512]), i] for ex in data]


        else:
            politics = ['18828_talk.politics.guns', '18828_talk.politics.mideast', '18828_talk.politics.misc'] # Label 0
            science = ['18828_sci.crypt', '18828_sci.electronics', '18828_sci.med', '18828_sci.space'] # 1
            religion = ['18828_alt.atheism', '18828_soc.religion.christian', '18828_talk.religion.misc'] # 2
            computer = ['18828_comp.graphics', '18828_comp.os.ms-windows.misc', '18828_comp.sys.ibm.pc.hardware',
            '18828_comp.sys.mac.hardware', '18828_comp.windows.x'] # 3
            sports = ['18828_rec.autos', '18828_rec.motorcycles', '18828_rec.sport.baseball', '18828_rec.sport.hockey'] # 4
            sale = ['18828_misc.forsale'] # 5

            for i, category in enumerate([politics, science, religion, computer, sports, sale]):
                for subcat in category:
                    data = load_dataset("newsgroup", subcat)['train']['text']
                    text += [[" ".join(ex.split('\n', 2)[2].split()[:512]), i] for ex in data]


        train_val, test = train_test_split(text, test_size=0.2, random_state=42)
        train, val = train_test_split(train_val, test_size=0.1, random_state=42)
        train = self.process_ng(train[:conf.sample], tokenizers[conf.tokenizer])
        val = self.process_ng(val[:conf.sample], tokenizers[conf.tokenizer])
        test = self.process_ng(test[:conf.sample], tokenizers[conf.tokenizer])

        return train, val, test


    def process_ng(self, data, tokenizer):
        """ Extracts the headlines and labels from the 20 newsgroups dataset. """
        text = [b[0] for b in data]
        text = tokenizer(text, padding=True, return_tensors='pt', truncation=True, max_length=512)['input_ids']

        labels = [b[1] for b in data]
        labels = torch.LongTensor(labels)

        return [{"txt" : h, "label" : l} for h, l in zip(text, labels)]


    def load_dbpedia(self, conf):
        dataset = load_dataset('dbpedia_14')
        train, val = train_test_split(dataset['train'], test_size=0.1, random_state=42)
        test = dataset['test']
        sample = conf.sample
        train = self.process_dbpedia(train, tokenizers[conf.tokenizer], sample)
        val = self.process_dbpedia(val, tokenizers[conf.tokenizer], sample)
        test = self.process_dbpedia(test, tokenizers[conf.tokenizer], sample)

        return train, val, test


    def process_dbpedia(self, data, tokenizer, sample=None, max_text_length=-1):
        text = [title + ' ' + content for (title, content) in zip(data['title'][:sample], data['content'][:sample])]
        text = tokenizer(text, padding=True, return_tensors='pt', truncation=True, max_length=512)['input_ids']

        labels = [label for label in data['label'][:sample]]
        labels = torch.LongTensor(labels)

        return [{"txt" : h, "label" : l} for h, l in zip(text, labels)]


    def load_datasets(self, conf):
        self.train = {}
        self.val = {}
        self.test = {}

        datasets = conf.train_sets.split(',')

        for dataset in datasets:
            if dataset == "hp":
                train, val, test = self.load_hp(conf)
            elif dataset == "bbc":
                train, val, test = self.load_bbc(conf)
            elif dataset == "ag":
                train, val, test = self.load_ag(conf)
            elif dataset == "dbpedia":
                train, val, test = self.load_dbpedia(conf)
            elif dataset == "ng":
                train, val, test = self.load_ng(conf)

            self.train[dataset] = train
            self.val[dataset] = val
            self.test[dataset] = test

class MergeMultitaskData(data.Dataset):
    def __init__(self, dataset_dict):
        super().__init__()
        self.datasets = dataset_dict
        self.dataset_names = list(dataset_dict.keys())
        self.lengths = {}
        for dataset in self.dataset_names:
            self.lengths[dataset] = len(self.datasets[dataset])

    def __len__(self):
        return sum([self.lengths[dataset] for dataset in self.datasets])

    def __getitem__(self, idx):
        batch = {}
        for dataset in self.dataset_names:
            batch[dataset] = self.datasets[dataset][idx%self.lengths[dataset]]

        return batch


class Args():
    def __init__(self):
        self.path = "models/bert"
        self.optimizer = "Adam"
        self.lr = 0.001
        self.max_epochs = 100
        self.finetuned_layers = 0
        self.tokenizer = "BERT"
        self.batch_size = 64
        self.device = "gpu"
        self.seed = 20
        self.max_text_length = -1
        self.sample = 100
        self.train_sets = "hp,ag,dbpedia"

if __name__ == "__main__":
    conf = Args()
    multitask_data = LoadMultitaskData(conf)
    train_data = MergeMultitaskData(multitask_data.train)
    loader = data.DataLoader(train_data, batch_size = conf.batch_size)
    batch = next(iter(loader))
    print(batch)
