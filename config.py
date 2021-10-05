from argparse import ArgumentParser
from re import M
import os

arg_defaults = {
    "path" : os.path.join("models", "bert_multitask"),
    "data_path" : os.path.join('.', 'Data'),
    "cache_path" : os.path.join('.', 'Cache'),
    "optimizer" : "Adam",
    "lr" : 0.001,
    "max_epochs" : 10,
    "finetuned_layers" : 0,
    "task_layers" : 1,
    "tokenizer" : "BERT",
    "batch_size" : 64,
    "device" : "cuda",
    "seed" : 20,
    "max_text_length": -1,
    "sample" : None,
    'progress_bar' : True,
    "num_workers" : 4,
    "gpus": 1,
    "train_sets" : "hp,ng,dbpedia",
    "test_set" : 'bbc',
    "hidden" : 192,
    "max_tokens" : 256,
    "inner_steps" : 5,
    "inner_lr" : 0.001,
    "shot" : 5,
    "eval_perm" : 3,
    "eval_way" : 4,
    "class_eval" : 32,
    "query_batch": 10
}

def get_args():
    parser = ArgumentParser(description="BERT baseline training")
    help_text_default = " (default: {})"
    parser.add_argument("name", type=str,
                        help="name of the model")
    parser.add_argument("--path", type=str, default=arg_defaults["path"],
                        help="the path to save the model checkpoints and logs"  + help_text_default.format(arg_defaults["path"]))
    parser.add_argument("--optimizer", type=str, default=arg_defaults["optimizer"],
                        choices=["Adam", "SGD"], help="the optimizer to use for training"  + help_text_default.format(arg_defaults["optimizer"]))
    parser.add_argument("--lr", type=float, default=arg_defaults["lr"],
                        help="the learning rate for the optimizer"  + help_text_default.format(arg_defaults["lr"]))
    parser.add_argument("--max_epochs", type=int, default=arg_defaults["max_epochs"],
                        help="the number of epochs after which to stop"  + help_text_default.format(arg_defaults["max_epochs"]))
    parser.add_argument("--finetuned_layers", type=int, default=arg_defaults["finetuned_layers"],
                        help="the number of transformer layers of BERT to finetune (-1: all layers)"  + help_text_default.format(arg_defaults["finetuned_layers"]))
    parser.add_argument("--task_layers", type=int, default=arg_defaults["task_layers"],
                        help="the number of task-specific layers of BERT in the multitask setup"  + help_text_default.format(arg_defaults["task_layers"]))
    parser.add_argument("--tokenizer", type=str, default=arg_defaults["tokenizer"],
                        choices=["BERT"], help="the tokenizer to use on the text"  + help_text_default.format(arg_defaults["tokenizer"]))
    parser.add_argument("--batch_size", type=int, default=arg_defaults["batch_size"],
                        help="size of the batches"  + help_text_default.format(arg_defaults["batch_size"]))
    parser.add_argument("--device", type=str, default=arg_defaults["device"],
                        choices=["cpu", "cuda"], help="the device to use"  + help_text_default.format(arg_defaults["device"]))
    parser.add_argument("--seed", type=int, default=arg_defaults["seed"],
                        help="the random seed used by pytorch lightning"  + help_text_default.format(arg_defaults["seed"]))
    parser.add_argument("--progress_bar", action="store_true", default=False,
                        help="show the progress bar")
    parser.add_argument("--max_text_length", type=int, default=arg_defaults["max_text_length"],
                        help="the max text length in characters (-1: no limit)"  + help_text_default.format(arg_defaults["max_text_length"]))
    parser.add_argument("--sample", type=int, default=arg_defaults["sample"],
                        help="Amount of datapoints used in each split in a dataset. Recommended for testing." + help_text_default.format(arg_defaults["sample"]))
    parser.add_argument("--num_workers", type=int, default=arg_defaults['num_workers'],
                        help="The number of workers/processes on the cpu" + help_text_default.format(arg_defaults["num_workers"]))
    parser.add_argument("--gpus", type=int, default=arg_defaults['gpus'],
                        help="The number of GPU's used" + help_text_default.format(arg_defaults["gpus"]))
    parser.add_argument("--train_sets", type=str, default=arg_defaults['train_sets'],
                        help="Comma separated names of datasets used to train multitask model." + help_text_default.format(arg_defaults['train_sets']))
    parser.add_argument("--test_set", type=str, default=arg_defaults['test_set'],
                        help="name of dataset used for few shot evaluation" + help_text_default.format(arg_defaults['test_set']))
    parser.add_argument("--hidden", type=int, default=arg_defaults['hidden'],
                        help="number of hidden units of the few shot head" + help_text_default.format(arg_defaults['hidden']))
    parser.add_argument("--max_tokens", type=int, default=arg_defaults['max_tokens'],
                        help="maximum number of tokens for the bert tokenizer (used when building data for few shot evaluation)" + help_text_default.format(arg_defaults['max_tokens']))
    parser.add_argument("--inner_steps", type=int, default=arg_defaults['inner_steps'],
                        help="number of steps in inner loop" + help_text_default.format(arg_defaults['inner_steps']))
    parser.add_argument("--inner_lr", type=int, default=arg_defaults['inner_lr'],
                        help="learning rate of inner loop" + help_text_default.format(arg_defaults['inner_lr']))
    parser.add_argument("--shot", type=int, default=arg_defaults['shot'],
                        help="number of shots in few shot evaluation" + help_text_default.format(arg_defaults['shot']))
    parser.add_argument("--query_batch", type=int, default=arg_defaults['query_batch'],
                        help="size of query batches" + help_text_default.format(arg_defaults['query_batch']))
    parser.add_argument("--eval_perm", type=int, default=arg_defaults['eval_perm'],
                        help="number of support sets in few shot evaluation" + help_text_default.format(arg_defaults['eval_perm']))
    parser.add_argument("--eval_way", type=int, default=arg_defaults['eval_way'],
                        help="number of 'ways' in few shot evaluation" + help_text_default.format(arg_defaults['eval_way']))
    parser.add_argument("--class_eval", type=int, default=arg_defaults['class_eval'],
                        help="amount of query examples per class in few shot evaluation" + help_text_default.format(arg_defaults['class_eval']))
    parser.add_argument("--data_path", type=str, default=arg_defaults['data_path'],
                        help="path to store data for few shot evaluation" + help_text_default.format(arg_defaults['data_path']))
    parser.add_argument("--cache_path", type=str, default=arg_defaults['cache_path'],
                        help="path to store data temporarily" + help_text_default.format(arg_defaults['cache_path']))

    args = parser.parse_args()
    return args
