# INSA Lyon GE 2021 - TDSI
# Shoulder Muscle Segmentation

## Install nnUNet package
- Remove old nnUNet
- Clone this repo
- Activate your PyTorch environment
- Run:
```
cd TDSI21-Shoulder-Muscle-Segmentation
pip install -e .
```

To use RMSProp : 
```
nnUNet_train 2d rmsPropTrainer Task500_Epaule FOLD --npz
```
## TransUNet
- Download Google pre-trained ViT models

    Get models in this link: R50-ViT-B_16, ViT-B_16, ViT-L_16...

```
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz &&
mkdir ../model/vit_checkpoint/imagenet21k &&
mv {MODEL_NAME}.npz ../model/vit_checkpoint/imagenet21k/{MODEL_NAME}.npz
```