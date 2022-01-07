import numpy as np
import torch
import torch.nn as nn
from nnunet.network_architecture.neural_network import SegmentationNetwork
from ml_collections import ConfigDict
from nnunet.network_architecture.transunet.vit_seg_modeling import VisionTransformer

class GenericTransUNet(VisionTransformer, SegmentationNetwork):

    use_this_for_batch_size_computation_2D = 1684522240.0 # default R50-ViT-B16, (1, 1, 384, 384)

    def __init__(self, config: ConfigDict, img_size=384, do_ds=False, **kwargs):
        super().__init__(config=config, img_size=img_size)
        self.conv_op = nn.Conv2d
        self.num_classes = config.n_classes
        self.do_ds = do_ds

    @staticmethod
    def compute_approx_vram_consumption(config, batch_size):
        """Naively estimate vram consumption
        """
        return GenericTransUNet.use_this_for_batch_size_computation_2D * batch_size * 1.2

# for development test
if __name__ == "__main__":
    from nnunet.network_architecture.transunet.vit_seg_configs import get_r50_b16_config
    config = get_r50_b16_config(6)
    config.img_size = 384
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = GenericTransUNet(config, do_ds=True).to(device)
    x = torch.rand(1, 1, config.img_size, config.img_size).to(device)
    # print(model)
    y = model(x)
    print("hold")