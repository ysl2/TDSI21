import numpy as np
import torch
import torch.nn as nn
from nnunet.network_architecture.neural_network import SegmentationNetwork
from ml_collections import ConfigDict
from nnunet.network_architecture.transunet.vit_seg_modeling_3d import VisionTransformer3D

# Not so generic after all !
class GenericTransUNet3D(VisionTransformer3D, SegmentationNetwork):

    use_this_for_batch_size_computation_3D = 12182355968.0 # default R50-ViT-B16, (2, 1, 96, 192, 192)

    def __init__(self, config: ConfigDict, img_size=384, do_ds=False, **kwargs):
        super().__init__(config=config, img_size=img_size)
        self.conv_op = nn.Conv3d
        self.num_classes = config.n_classes
        self.do_ds = do_ds

    @staticmethod
    def compute_approx_vram_consumption(config, batch_size):
        """Naively estimate vram consumption
        """
        return GenericTransUNet3D.use_this_for_batch_size_computation_3D * (batch_size // 2)

# for development test
if __name__ == "__main__":
    from nnunet.network_architecture.transunet.vit_seg_configs import get_r50_b16_3d_config
    config = get_r50_b16_3d_config(6)
    config.img_size = (96, 192, 192)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = GenericTransUNet3D(config, do_ds=True).to(device)
    x = torch.rand(2, 1, *config.img_size).to(device)
    # print(model)
    y = model(x)
    print("hold")