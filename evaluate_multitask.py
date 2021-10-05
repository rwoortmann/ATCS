import torch
import os
from torch.utils import data
import torch.optim as optim

from multitask_data import LoadMultitaskData, MergeMultitaskData
from multitask_model import MultitaskBert
from multitask_trainer import MultitaskTrainer
import proto_data
import proto_utils

from config import get_args

def evaluate_multitask(conf, test_data, model, optimizer):
    print("Testing multitask model...")
    checkpoint = torch.load(f'{conf.path}/checkpoints/{conf.name}.ckpt', map_location=torch.device(conf.device))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    batch = proto_utils.get_test_batch(conf, test_data)
    test_acc, test_loss = model.eval_model(batch)
    print("Test acc", test_acc.item(), "Test loss", test_loss.item())
    return None

if __name__ == "__main__":
    conf = get_args()
    print("-------------- CONF ---------------")
    print(conf)
    print("-----------------------------------")

    test_data = proto_data.DataLoader(conf, conf.test_set)
    model = MultitaskTrainer(conf).to(conf.device)
    if conf.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=conf.lr)
    elif conf.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=conf.lr)
    evaluate_multitask(conf, test_data, model, optimizer)
