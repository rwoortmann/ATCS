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
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def train_multitask(conf, train_data, test_data, writer):
    model = MultitaskTrainer(conf).to(conf.device)
    os.makedirs(f'{conf.path}/checkpoints/', exist_ok=True)

    if conf.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=conf.lr)
    elif conf.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=conf.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 6], gamma=0.1)
    best_val_acc = 0

    for epoch in tqdm(range(conf.max_epochs), desc="Epoch"):
        train_acc, train_loss = model.train_model(train_data, optimizer, writer)
        writer.add_scalar("Train accuracy", train_acc.item(), epoch)
        if epoch % 5 == 0:
            batch = proto_utils.get_test_batch(conf, test_data)
            val_acc, val_loss = model.eval_model(batch)
            writer.add_scalar("Val loss", val_loss.item(), epoch)
            writer.add_scalar("Val accuracy", val_acc.item(), epoch)
            if val_acc > best_val_acc:
                torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            }, f'{conf.path}/checkpoints/{conf.name}.ckpt')
        scheduler.step()

    print("Testing multitask model...")
    checkpoint = torch.load(f'{conf.path}/checkpoints/{conf.name}.ckpt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    batch = proto_utils.get_test_batch(conf, test_data)
    test_acc, test_loss = model.eval_model(batch)
    print("Test acc", test_acc, "Test loss", test_loss)
    writer.add_scalar("Test loss", test_loss.item())
    writer.add_scalar("Test accuracy", test_acc.item())
    writer.flush()
    writer.close()


if __name__ == "__main__":
    conf = get_args()
    print("-------------- CONF ---------------")
    print(conf)
    print("-----------------------------------")

    print("Loading data...")
    multitask_data = LoadMultitaskData(conf)
    multitask_train = MergeMultitaskData(multitask_data.train)
    train_data = data.DataLoader(multitask_train, batch_size=conf.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=conf.num_workers) if multitask_train != None else None
    test_data = proto_data.DataLoader(conf, conf.test_set)

    print("Training multitask model...")
    save_name = f'./{conf.path}/runs/{conf.name}'
    writer = SummaryWriter(save_name)
    train_multitask(conf, train_data, test_data, writer)
