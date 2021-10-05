from datasets import load_dataset
from torch.utils.data import Dataset
from operator import itemgetter
import random
import torch
import numpy as np
import os
import shutil
import csv
import proto_utils


class Episode():
    def __init__(self, n_classes, support, query):
        self.n_classes = n_classes
        self.support = self.Set(support, n_classes)
        self.query = self.Set(query, n_classes)
        self.query.shuffle()

    class Set():
        def __init__(self, text, n_classes):
            self.text = np.array(text)
            self.n_classes = n_classes
            self.len = len(text)
            self.labels = torch.flatten(torch.tensor([[i] * int(self.len/n_classes) for i in range(n_classes)]))
            self.idx = torch.arange(self.len)

        def shuffle(self):
            self.idx = self.idx[torch.randperm(self.idx.shape[0])]

        def get_batch(self, config, batch_size=-1, tokenize=True, shuffle=False):
            batch_size = self.len if batch_size == -1 else batch_size
            if shuffle: self.shuffle()

            for i in range(0, self.len, batch_size):
                tb = self.text[self.idx[i:min(i + batch_size, self.len)]].tolist()
                lb = self.labels[self.idx[i:min(i + batch_size, self.len)]]
                if tokenize == True:
                    tb = proto_utils.tokenize(config, tb)

                yield tb.to(config.device), lb.to(config.device)





# data[class_id] = [sentences]
class DataLoader():
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset
        self.set_path = os.path.join(config.data_path, dataset)
        self.task = self.TaskClass()

        if dataset == "hp":#200k
            self.init_data(self.process_hp)
        if dataset == "bbc":#1490
            self.init_data(self.process_bbc)
        if dataset == "ag":#1M
            self.init_data(self.process_ag)
        if dataset == "yahoo":#1M
            self.init_data(self.process_yahoo)
        if dataset == "dbpedia":#40k
            self.init_data(self.process_dbpedia)
        if dataset == "ng":
            self.init_data(self.process_ng)


    class TaskClass():
        def __init__(self):
            self.name = ""
            self.data = {}

        def sample_episode(self, config):
            way = random.randint(config.min_way, min(config.max_way, len(self.data))) # amount of ways, min_ways <= n <= min(max_ways, n_classes)
            class_ids = random.sample(range(len(self.data)), way) # sample classes, n = way
            query, support = [], []

            for class_id in class_ids: # for all classes
                sample_ids = random.sample(range(len(self.data[class_id])), config.query_size + config.shot) # number of sampled ids = n_query + n_support
                support.extend([self.data[class_id][sample_id] for sample_id in sample_ids[:config.shot]]) # class_support = first sampled ids, n = shot
                query.extend([self.data[class_id][sample_id] for sample_id in sample_ids[config.shot:]]) # class_query = last sampled ids, n = query_size

            return Episode(way, support, query)

        def sample_eval_episode(self, config):
            way = min(config.eval_way, len(self.data))
            class_eval = config.class_eval
            if config.class_eval == -1:
                class_eval = min([len(self.data[data_class]) for data_class in self.data]) - config.shot

            class_ids = random.sample(range(len(self.data)), way)
            query, support = [], []

            for class_id in class_ids:
                sample_ids = random.sample(range(len(self.data[class_id])), class_eval + config.shot) #class_eval != query_size
                support.extend([self.data[class_id][sample_id] for sample_id in sample_ids[:config.shot]])
                query.extend([self.data[class_id][sample_id] for sample_id in sample_ids[config.shot:]])

            return Episode(way, support, query)


    # Download, process, save, remove raw dataset
    def init_data(self, process_fn):
        if os.path.exists(self.set_path):
            self.load_data()
        else:
            os.mkdir(self.config.cache_path)
            process_fn()
            shutil.rmtree(self.config.cache_path) # remove cache


    def load_data(self):
        self.task = torch.load(os.path.join(self.set_path, 'train.pt'))

    def save_data(self, name, data):
        self.task.name = name
        self.task.data = data
        os.mkdir(self.set_path)
        torch.save(self.task, os.path.join(self.set_path, 'train.pt'))


    """DATASET SPECIFIC"""
    def process_hp(self):
        data = {i: [] for i in range(41)}
        dataset = load_dataset('Fraser/news-category-dataset', split='train', cache_dir=self.config.cache_path)
        for example in dataset:
            text = example['headline'] + ' ' + example['short_description']
            data[example['category_num']].append(text)
        self.save_data('hp', data)


    def process_ag(self):
        data = {i: [] for i in range(4)}
        dataset = load_dataset('ag_news', split='train', cache_dir=self.config.cache_path)
        for example in dataset:
            text = example['text']
            text = text.replace('\\', ' ')
            data[example['label']].append(text)
        self.save_data('ag', data)


    def process_yahoo(self):
        data = {i: [] for i in range(10)}
        dataset = load_dataset('yahoo_answers_topics', split='train', cache_dir=self.config.cache_path)
        for example in dataset:
            text = example['question_title'] + ' ' + example['question_content'] + ' ' + example['best_answer']
            #text = text.replace('\n', '')
            data[example['topic']].append(text)
        self.save_data('yahoo', data)


    def process_dbpedia(self):
        data = {i: [] for i in range(14)}
        dataset = load_dataset('dbpedia_14', split='train', cache_dir=self.config.cache_path)
        for example in dataset:
            text = example['title'] + ' ' + example['content']
            data[example['label']].append(text)
        self.save_data('dbpedia', data)


    def process_bbc(self):
        data = {i: [] for i in range(5)}
        cats = {'business': 0, 'entertainment': 1, 'politics': 2, 'sport': 3, 'tech': 4}
        with open(os.path.join('Data/BBC/Train/BBC News Train.csv')) as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)
            for row in reader:
                text = row[1].replace('\n', ' ')
                data[cats[row[2]]].append(text)
        self.save_data('bbc', data)

    def process_ng(self):
        data = {i: [] for i in range(20)}
        cats = {'18828_alt.atheism': 0, '18828_comp.graphics': 1, '18828_comp.os.ms-windows.misc': 2, '18828_comp.sys.ibm.pc.hardware': 3,
            '18828_comp.sys.mac.hardware': 4, '18828_comp.windows.x': 5, '18828_misc.forsale': 6, '18828_rec.autos': 7, '18828_rec.motorcycles': 8,
            '18828_rec.sport.baseball': 9, '18828_rec.sport.hockey': 10, '18828_sci.crypt': 11, '18828_sci.electronics':  12, '18828_sci.med': 13, '18828_sci.space': 14,
            '18828_soc.religion.christian': 15, '18828_talk.politics.guns': 16, '18828_talk.politics.mideast': 17, '18828_talk.politics.misc': 18,
            '18828_talk.religion.misc': 19}
        for (cat, label) in cats.items():
            dataset = load_dataset('newsgroup', cat, split='train', cache_dir=self.config.cache_path)
            for example in dataset:
                text = example['text'][20:].replace('\n', ' ')
                data[label].append(text)
        self.save_data('ng', data)
