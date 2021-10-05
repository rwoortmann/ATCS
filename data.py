import torch
from transformers import BertTokenizer
from datasets import load_dataset

from sklearn.model_selection import train_test_split

tokenizers = {
    "BERT" : BertTokenizer.from_pretrained('bert-base-uncased')
}

class Dataset():
    def __init__(self, conf):
        if conf.dataset == "hp":
            self.load_hp(conf)
        elif conf.dataset == "ag":
            self.load_ag(conf)
        elif conf.dataset == "bbc":
            self.load_bcc(conf)
        elif conf.dataset == "ng":
            self.load_ng(conf)
        elif conf.dataset == "db":
            self.load_db(conf)
        else:
            pass

    def load_hp(self, conf):
        """ Loads the HuffPost (hp) dataset from [Hugging Face](https://huggingface.co/datasets/Fraser/news-category-dataset) 
            Splits in train (160682), validation (10043) and test (30128). Every sample has the following features:
            `['authors', 'category', 'category_num', 'date', 'headline', 'link', 'short_description']` """
        dataset = load_dataset("Fraser/news-category-dataset")
        self.train = self.process_hp(dataset["train"], tokenizers[conf.tokenizer], conf.max_text_length)
        self.val = self.process_hp(dataset["validation"], tokenizers[conf.tokenizer], conf.max_text_length)
        self.test = self.process_hp(dataset["test"], tokenizers[conf.tokenizer], conf.max_text_length)

    def process_hp(self, data, tokenizer, max_text_length=-1):
        """ Extracts the headlines and labels from the HuffPost dataset.
            The description is empty for some samples, which is why it is not returned."""
        txt = [head + ' ' + desc for head, desc in zip(data["headline"], data["short_description"])]
        maxlength = [ex[:max_text_length] for ex in txt]
        txt = tokenizer(maxlength, padding=True, return_tensors='pt')["input_ids"]
        labels = torch.LongTensor(data["category_num"])
        return [{"txt" : h, "label" : l} for h, l in zip(txt, labels)]

    def load_ag(self, conf):
        """ Loads the AG news dataset from [Hugging Face](https://huggingface.co/datasets/ag_news) 
            Splits in train (120000) and test (7600). Every sample has the following features: `['label', 'text']` """
        dataset = load_dataset("ag_news")
        self.train = self.process_ag(dataset["train"], tokenizers[conf.tokenizer], conf.max_text_length)
        self.train, self.val = train_test_split(self.train, test_size=0.2, random_state=42)
        self.test = self.process_ag(dataset["test"], tokenizers[conf.tokenizer], conf.max_text_length)

    def process_ag(self, data, tokenizer, max_text_length=-1):
        """ Extracts the headlines and labels from the AG news dataset. 
            The headlines contain '\\' characters in place of newlines. """
        maxlength = [ex[:max_text_length] for ex in data["text"]]
        noslash = [ex.replace("\\", " ") for ex in maxlength]
        headlines = tokenizer(noslash, padding=True, return_tensors='pt')["input_ids"]
        labels = torch.LongTensor(data["label"])
        return [{"txt" : h, "label" : l} for h, l in zip(headlines, labels)]

    def load_bcc(self, conf):
        """ Loads the BBC news dataset from [Kaggle](https://www.kaggle.com/c/learn-ai-bbc) 
            This dataset has to be downloaded manually using the `downloadbbcdata` file. 
            Splits in train (1490) and test (735)."""
        l2i = {'entertainment':0, 'business':1, 'politics':2, 'sport':3, 'tech':4}
        # first row are colum names: ['ArticleId' 'Text' 'Category']
        train = [line.replace("\n", "").split(",") for line in open('Data/BBC/Train/BBC News Train.csv')][1:]
        train = [[ex[1], l2i[ex[2]]] for ex in train]
        
        # first row are colum names: ['ArticleId' 'Text']
        test_ex = [line.replace("\n", "").split(",") for line in open('Data/BBC/Test/BBC News Test.csv')][1:]
        # first row are colum names: ['ArticleId' 'Category']
        test_labels = [line.replace("\n", "").split(",") for line in open('Data/BBC/Test/BBC News Sample Solution.csv')][1:]
        test = [[ex[1], l2i[label[1]]] for ex, label in zip(test_ex, test_labels) if ex[0] == label[0]]

        self.train = self.process_bbc(train, tokenizers[conf.tokenizer], conf.max_text_length)
        self.train, self.val = train_test_split(self.train, test_size=0.2, random_state=42)
        self.test = self.process_bbc(test, tokenizers[conf.tokenizer], conf.max_text_length)

        assert len(self.test) == len(test_labels)
        "The BBC test datafiles were corrupted!"

    def process_bbc(self, data, tokenizer, max_text_length=-1):
        """ Extracts the headlines and labels from the BBC news dataset. """
        maxlenght = [b[0][:max_text_length] for b in data]
        headlines = tokenizer(maxlenght, padding=True, return_tensors='pt')["input_ids"]

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
        self.train = self.process_ng(train, tokenizers[conf.tokenizer], conf.max_text_length)
        self.val = self.process_ng(val, tokenizers[conf.tokenizer], conf.max_text_length)
        self.test = self.process_ng(test, tokenizers[conf.tokenizer], conf.max_text_length)


    def process_ng(self, data, tokenizer, max_text_length=-1):
        """ Extracts the headlines and labels from the 20 newsgroups dataset. """
        text = [b[0] for b in data]
        text = tokenizer(text, padding=True, return_tensors='pt', truncation=True, max_length=512)["input_ids"]
        labels = torch.LongTensor([b[1] for b in data])

        return [{"txt" : h, "label" : l} for h, l in zip(text, labels)]

    def load_db(self, conf):
        """ Loads the dbpedia (db) """
        dataset = load_dataset('dbpedia_14')
        train, val = train_test_split(dataset["train"], test_size=0.1, random_state=42)
        self.train = self.process_db(train, tokenizers[conf.tokenizer], conf.max_text_length)
        self.val = self.process_db(val, tokenizers[conf.tokenizer], conf.max_text_length)
        self.test = self.process_db(dataset["test"], tokenizers[conf.tokenizer], conf.max_text_length)
        

    def process_db(self, data, tokenizer, max_text_length=-1):
        """ Extracts the headlines and labels from the DBPEDIA dataset.
            The description is empty for some samples, which is why it is not returned."""  
        txt = [title + ' ' + cont for title, cont in zip(data["title"], data["content"])]
        maxlength = [ex[:max_text_length] for ex in txt]
        txt = tokenizer(maxlength, padding=True, return_tensors='pt', truncation=True, max_length=512)['input_ids']
        labels = torch.LongTensor(data["label"])
        return [{"txt" : h, "label" : l} for h, l in zip(txt, labels)]