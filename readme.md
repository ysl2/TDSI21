# MRI shoulder muscle segmentation using nnUNet

By [Clément Nicolas--Graffard](https://www.linkedin.com/in/cl%C3%A9ment-nicolas--graffard/), [Guillaume Picaud](https://www.linkedin.com/in/guillaume-picaud-27754a1ba/) and [Ngoc Trong Nghia Nguyen](https://www.linkedin.com/in/ngoc-trong-nghia-nguyen/) of INSA Lyon - 5GE - TDSI - 2021.

This is an academic project of TDSI program (Image and Signal Processing) at INSA Lyon under the supervision of Professor [Thomas Grenier, Ph.D](https://www.creatis.insa-lyon.fr/~grenier/).

The original README of nnUNet can be found [here](/nnunet_readme.md)

Our interpretation of nnUNet is resumed in the Wiki pages of this repository. --> see [here](https://github.com/nntrongnghia/TDSI21-Shoulder-Muscle-Segmentation/wiki)

# Install this nnUNet package
- Remove old nnUNet
- Clone this repo
- Activate your PyTorch environment
- Install nnUNet in this repository:

```
cd TDSI21-Shoulder-Muscle-Segmentation
pip install -e .
```

# New features added

## RMSProp 
To use RMSProp in training: 
```
nnUNet_train MODEL_NAME rmsPropTrainer Task500_Epaule FOLD --npz
```

## TransUNet
TransUNet for 2D images is added in this nnUNet using the code in [the official repository](https://github.com/Beckschen/TransUNet).
Then we add our 3D version by adapting 2D convolution layers to 3D.

For more detail about training, inference and evaluation, please check [Wiki/TransUNet 2D & 3D](../../wiki/TransUNet-2D-&-3D)

## Model summary

Once the training is done, you can check the number of parameters, intermediate tensor shape and model structure by `nnUNet_model_summary`
```
nnUNet_model_summary -t TASK_NAME -m MODEL_NAME
```

## Minor modifications
- Early stopping with patien of 50 epochs
- Elastic Deformation in data augmentation is available through `--do_elastic`
- Advanced metrics for evaluation: Hausdorff Distance and Average Symmetric Surface Distance 

