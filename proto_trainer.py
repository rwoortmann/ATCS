import torch
import itertools
import copy
import torch.nn as nn
import torch.optim as optim
import proto_utils
from proto_classifier import BertClassifier
from tqdm import tqdm


class ProtoTrainer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = BertClassifier(config)
        self.loss_module = nn.CrossEntropyLoss()


    def meta_train(self, batch, outer_opt):
        accs, losses = [], []
        outer_opt.zero_grad()

        for episode in batch:
            episode_model, weight, bias = self.proto_task(episode.support) # create task specific model

            for (text, labels) in episode.query.get_batch(self.config, batch_size=self.config.query_batch):

                out = episode_model(text)
                out = self.output_layer(out, weight, bias)
                loss = self.loss_module(out, labels)

                proto_grad = torch.autograd.grad(loss, [p for p in self.model.parameters() if p.requires_grad], retain_graph=True)
                meta_grad = torch.autograd.grad(loss, [p for p in episode_model.parameters() if p.requires_grad])
                grads = [mg + pg for (mg, pg) in zip(meta_grad, proto_grad)]

                for param, grad in zip([p for p in self.model.parameters() if p.requires_grad], grads):
                    if param.grad == None:
                        param.grad = grad.detach()
                    else:
                        param.grad += grad.detach()

                # logging
                accs.extend((out.argmax(dim=-1) == labels).float().detach().cpu())
                losses.append(loss.item())

        for param in [p for p in self.model.parameters() if p.requires_grad]:
            param.grad /= len(losses)
        outer_opt.step()

        train_acc = torch.mean(torch.tensor(accs)).cpu()
        train_loss = torch.mean(torch.tensor(losses)).cpu()

        return train_acc, train_loss


    def eval_model(self, batch):
        accs, losses = [], []

        for episode in batch:
            episode_model, weight, bias = self.proto_task(episode.support)
            with torch.no_grad():
                for (text, labels) in episode.query.get_batch(self.config, batch_size=self.config.query_batch):
                    out = episode_model(text)
                    out = self.output_layer(out, weight, bias)
                    loss = self.loss_module(out, labels)
                    losses.append(loss.item())
                    accs.extend((out.argmax(dim=-1) == labels).float())

        eval_acc = torch.mean(torch.tensor(accs))
        eval_loss = torch.mean(torch.tensor(losses))

        return eval_acc, eval_loss



    def proto_task(self, support):
        episode_model = copy.deepcopy(self.model)
        episode_model.zero_grad()

        # init prototype parameters
        proto_W, proto_b = self.prototype(support)
        weight = proto_W.clone().detach().requires_grad_(True)
        bias = proto_b.clone().detach().requires_grad_(True)

        params = [p for p in episode_model.parameters() if p.requires_grad] + [weight, bias]
        #params = [weight, bias]
        inner_opt = torch.optim.SGD(params, lr=self.config.inner_lr)

        # inner loop
        for i in range(self.config.inner_steps):

            inner_opt.zero_grad()
            for j, (text, labels) in enumerate(support.get_batch(self.config, batch_size=self.config.sup_batch)):
                out = episode_model(text)
                out = self.output_layer(out, weight, bias)
                loss = self.loss_module(out, labels)
                loss.backward(inputs=params)
            inner_opt.step()
        inner_opt.zero_grad()

        # add prototypes back to the computation graph
        weight = 2 * proto_W + (weight - 2 * proto_W).detach()
        bias = 2 * proto_b + (bias - 2 * proto_b).detach()

        return episode_model, weight, bias


    def prototype(self, support):
        sup_batch = support.len if self.config.sup_batch == -1 else self.config.sup_batch

        all_emb = torch.cat([self.model(text) for (text, labels) in support.get_batch(self.config, batch_size=sup_batch)])
        c_k = torch.stack([torch.mean(all_emb[i: i+self.config.shot, :], dim=0) for i in range(0, self.config.shot*support.n_classes, self.config.shot)])

        W = 2 * c_k
        b = -torch.linalg.norm(c_k, dim=1)**2
        support.shuffle()

        return W, b


    def output_layer(self, input, weight, bias):
        return torch.nn.functional.linear(input, weight, bias)
