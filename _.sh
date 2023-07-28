#!/bin/bash


nnUNet_dataset=/home/yusongli/Templates/yunet
export nnUNet_raw_data_base="${nnUNet_dataset}/nnUNet_raw"
export nnUNet_preprocessed="${nnUNet_dataset}/nnUNet_preprocessed"
export RESULTS_FOLDER="${nnUNet_dataset}/nnUNet_results"
export CUDA_DEVICE_ORDER=PCI_BUS_ID


TASK=002


function preprocess () {
    nnUNet_plan_and_preprocess \
        -t "${TASK}" \
        --planner3d ExperimentPlannerTransUNet3D \
        --planner2d ExperimentPlannerTransUNet
}


function train () {
    CUDA=0
    FOLD=0
    TRAINER=TransUNetTrainer_yusongli

    CUDA_VISIBLE_DEVICES=${CUDA} \
    nnUNet_train \
        TransUNet3D \
        "${TRAINER}" \
        "${TASK}" \
        "${FOLD}" \
        --npz
}


function main () {
    preprocess

    # FOLDS=(0 1 2 3 4)
    # for i in ${FOLDS[@]}; do
    #     train "${i}"
    # done

    # train
}


main
