#!/bin/bash


nnUNet_dataset=/home/yusongli/Templates/yunet
export nnUNet_raw_data_base="${nnUNet_dataset}/nnUNet_raw"
export nnUNet_preprocessed="${nnUNet_dataset}/nnUNet_preprocessed"
export RESULTS_FOLDER="${nnUNet_dataset}/nnUNet_results"
export CUDA_DEVICE_ORDER=PCI_BUS_ID


function preprocess () {
    task="$1"

    nnUNet_plan_and_preprocess \
        -t "$task" \
        --planner3d ExperimentPlannerTransUNet3D \
        --planner2d ExperimentPlannerTransUNet \
        -no_pp
}


function train () {
    cuda="$1"
    trainer="$2"
    task="$3"
    fold="$4"

    CUDA_VISIBLE_DEVICES=$cuda \
    nnUNet_train \
        TransUNet3D \
        "$trainer" \
        "$task" \
        "$fold" \
        --npz
}


function main () {
    # preprocess

    # FOLDS=(0 1 2 3 4)
    # for i in ${FOLDS[@]}; do
    #     train "${i}"
    # done

    trainer=TransUNetTrainer_yusongli
    task=203

    # preprocess "$task"
    train 5 "$trainer" "$task" 0 &
    train 6 "$trainer" "$task" 1 &
    train 7 "$trainer" "$task" 2 &
    train 5 "$trainer" "$task" 3 &
    train 6 "$trainer" "$task" 4 &
}


main
