import torch.nn as nn
import torch
import random
#random.seed(42)

def create_labels(way, shot):
    return torch.flatten(torch.tensor([[i] * shot for i in range(way)]))# labels are [0 x shot, ..., way x shot]

def tokenize(config, text):
    tok_text = config.tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=config.max_tokens)
    return tok_text

def get_train_batch(config, datasets):
    return [random.choice(datasets).task.sample_episode(config) for i in range(config.meta_batch)]

def get_train_batch2(config, datasets):
    return [dataset.task.sample_episode(config) for dataset in datasets]

def get_train_batch3(config, datasets):
    return [dataset.task.sample_episode(config) for dataset in datasets] + [dataset.task.sample_episode(config) for dataset in datasets]

def get_train_batch4(config, datasets):
    return [random.choice(datasets).task.sample_episode(config)]

def get_test_batch(config, dataset):
    return [dataset.task.sample_eval_episode(config) for i in range(config.eval_perm)]
