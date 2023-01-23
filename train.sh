#!/bin/bash
nnUNet_train \
TransUNet3D \
new_fine_transunet3d_epoch500 \
Task608_copy_607_for_TransUNet3D \
0 \
--npz \
--fp32
