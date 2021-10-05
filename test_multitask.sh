#!/bin/bash

# source activate atcs2

# 'run1_lr0.0001' is the name of the checkpoint file

# make sure to specify in the argument '--path' the path where
# the checkpoint file is store

sets="bbc ag"

for set in $sets
do
    echo $set

    for i in {1..20}
    do
        acc=`python evaluate_multitask.py run1_lr0.0001 --path lisa-models/bert_multitask --test_set=$set | grep "Test acc" | awk '{print $3}'`
        echo $acc
    done
done
