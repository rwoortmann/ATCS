# ACTS



The main folder directory contains:

* [checkpoint](https://drive.google.com/drive/folders/1B_iP5n9oyTLfqqp8guHk1gvFXM0guZb1?usp=sharing) folder containing pre-trained models and tensorboard logs

 To see tensorboard logs  `tensorboard --logdir checkpoints`
 
 * [SentEval](https://github.com/facebookresearch/SentEval) folder used in evaluation

In SentEval/data/downstream run `./get_transfer_data.bash` for transfer task data

* train.py
  Usage: `python train.py --encoder model (--BiLSTM_type max)`
  Where model can be AwE, LSTM or BiLSTM and when using BiLSTM its type can be max or simple
  
