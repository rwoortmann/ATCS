from argparse import ArgumentParser

arg_defaults = {
    "path" : "models/bert",
    "optimizer" : "Adam",
    "lr" : 0.001,
    "max_epochs" : 100,
    "finetuned_layers" : 0,
    "tokenizer" : "BERT",
    "batch_size" : 64,
    "device" : "gpu",
    "seed" : 20,
    "max_text_length": -1,
    "test_model": ""
}

help_text_default = " (default: {})"

def get_args():
    parser = ArgumentParser(description="BERT baseline training")

    parser.add_argument("name", type=str, 
                        help="name of the model")
    parser.add_argument("dataset", type=str, choices=["hp", "ag", "bbc", "ng", "db"],
                        help="the dataset used for training")
    parser.add_argument("nr_classes", type=int,
                        help="the number of classes of the dataset")
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
    parser.add_argument("--tokenizer", type=str, default=arg_defaults["tokenizer"],
                        choices=["BERT"], help="the tokenizer to use on the text"  + help_text_default.format(arg_defaults["tokenizer"]))
    parser.add_argument("--batch_size", type=int, default=arg_defaults["batch_size"],
                        help="size of the batches"  + help_text_default.format(arg_defaults["batch_size"]))
    parser.add_argument("--device", type=str, default=arg_defaults["device"],
                        choices=["cpu", "gpu"], help="the device to use"  + help_text_default.format(arg_defaults["device"]))
    parser.add_argument("--seed", type=int, default=arg_defaults["seed"],
                        help="the random seed used by pytorch lightning"  + help_text_default.format(arg_defaults["seed"]))
    parser.add_argument("--progress_bar", action="store_true", default=False,
                        help="show the progress bar")
    parser.add_argument("--max_text_length", type=int, default=arg_defaults["max_text_length"],
                        help="the max text length in characters (-1: no limit)"  + help_text_default.format(arg_defaults["max_text_length"]))

    parser.add_argument("--test_model", type=str, default=arg_defaults["test_model"],
                        help="path of the model to be tested")
    
    
    args = parser.parse_args()
    return args