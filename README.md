# ACTS



The main folder directory contains:

* [checkpoint](https://drive.google.com/drive/folders/1B_iP5n9oyTLfqqp8guHk1gvFXM0guZb1?usp=sharing) folder containing pre-trained models and tensorboard logs.

 To see tensorboard logs  `tensorboard --logdir checkpoints`
 
 * [SentEval](https://github.com/facebookresearch/SentEval) folder used in evaluation.

In SentEval/data/downstream run `./get_transfer_data.bash` for transfer task data.

* **train.py**
  Usage: `python train.py --encoder model (--BiLSTM_type max) (--savename ...)` 
  
  Where model can be: AwE, LSTM or BiLSTM 
  
  When using BiLSTM type can be: max or simple
  
  SNLI and GloVe data should be automatically downloaded and .data and .vector_cache respectively

* **eval.py**

  Usage: `python train.py --encoder model (--BiLSTM_type max)`
  
  using the same commands as train.py
  
  
 * **utils.py**

Contains various functions used in eval and the jupyter notebook.

* **snli_data.py**

For importing SNLI data and creating dataloaders.

* **infersent.py**
 
 Torch lighning class used for training and inference
 
* **InferSent_demo**

Notebook used to show results and demo inference

* **environement.yml**

File containing all packages needed to train and evaluate InferSent

* **report.pdf**
Pdf summarizing some of the findings and differences between this implementation and that of the original InferSent paper 
