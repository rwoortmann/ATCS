import torch
import proto_utils
import numpy as np

from transformers import BertTokenizer
from proto_data import DataLoader
from proto_trainer import ProtoTrainer
from torch.utils.tensorboard import SummaryWriter
from itertools import product
torch.manual_seed(42)

class config():
    def __init__(self):
        self.data_path = './Data/'
        self.cache_path = './Cache/'
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.seed = 42
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # bert h_parameters
        self.finetuned_layers = -1
        self.mlp_in = 768
        self.hidden = 192
        self.max_tokens = 256

         # maml h_parameters
        self.inner_steps = 5
        self.meta_batch = 4
        self.inner_lr = 0.0001 
        self.outer_lr = 0.01
        self.max_epochs = 251

        # episode h_parameters
        self.min_way = 4
        self.max_way = 5
        self.shot = 5
        self.query_size = 5
        
        # eval parameters
        self.eval_perm = 4 # evaluate model with 4 different support sets
        self.eval_way = 5
        self.class_eval = 32 # amount of query samples per class in eval

        self.query_batch = self.query_size # size of query batches
        self.sup_batch = self.shot

       

def train_protomaml(config, train_data, test_data, writer, get_best=False):
    
    model = ProtoTrainer(config).to(config.device)
    outer_opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=config.outer_lr)
    max_acc = 0
    
    for epoch in range(config.max_epochs):
        
        batch = proto_utils.get_train_batch2(config, train_data)
        t_acc, t_loss = model.meta_train(batch, outer_opt)
        writer.add_scalar("Train Loss", t_loss, epoch)
        writer.add_scalar("Train Accuracy", t_acc, epoch)

        if epoch%5 == 0:
            batch = proto_utils.get_test_batch(config, test_data)
            e_acc, e_loss = model.eval_model(batch)
            writer.add_scalar("Eval Loss", e_loss, epoch)
            writer.add_scalar("Eval Accuracy", e_acc, epoch)

            if get_best and e_acc > max_acc:
                max_acc = e_acc
                torch.save(model.state_dict(), './models/best_' + test_data.task.name)    

    # test on entire test set
    if get_best:
        test_best(config, model, train_data, test_data)


def test_best(config, model, train_data, test_data):
        config.eval_perm = 10
        config.class_eval = 5000
        if test_data.task.name == 'bbc':
            config.class_eval = -1
        
        batch = proto_utils.get_test_batch(config, test_data)
        t_acc, t_loss = model.eval_model(batch)
        writer.add_scalar("Test Accuracy", t_acc, 0)


def search_hyper_params(config):
    
    train_data = [DataLoader(config, set) for set in ['hp', 'ag', 'bbc', 'ng']]
    test_data = DataLoader(config, 'dbpedia')
    
    tune_params = dict(
        shot = [1, 5],
        inner_steps = [1, 5],
        inner_lr = [0.001, 0.0001])

    param_values = [v for v in tune_params.values()]
    for shot, inner_steps, inner_lr in product(*param_values):
        config.shot = shot
        config.inner_steps = inner_steps
        config.inner_lr = inner_lr
        
        #save_name = f'./runs/shot={shot}_inner_steps={inner_steps}_inner_lr={inner_lr}'
        save_name = './runs/shot=' + str(config.shot) + '_inner_steps=' + str(config.inner_steps)+ '_inner_lr=' + str(config.inner_lr) + '_db'
        writer = SummaryWriter(save_name)
        train_protomaml(config, train_data, test_data, writer, get_best=False)


if __name__ == "__main__":
    config = config()
    train_data = [DataLoader(config, set) for set in ['hp', 'dbpedia', 'ng']]
    
    #tuned parameters
    config.shot = 5
    config.inner_steps = 3
    config.inner_lr = 0.0001
     
    test_data = DataLoader(config, 'bbc')
    save_name = './runs/tuned_shot=' + str(config.shot) + '_bbc'
    writer = SummaryWriter(save_name)
    train_protomaml(config, train_data, test_data, writer, get_best=True)

    test_data = DataLoader(config, 'ag')
    save_name = './runs/tuned_shot=' + str(config.shot) + '_ag'
    writer = SummaryWriter(save_name)
    train_protomaml(config, train_data, test_data, writer, get_best=True)











