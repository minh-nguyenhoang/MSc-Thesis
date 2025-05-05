import torch 
import torch.nn as nn

from .iresnet import iresnet50, IResNet
from .metrics import ArcMarginProduct
from .neural_backdoor import ConfounderNet, SurrogateNet

from torch.optim import AdamW

class DeconfoundedModel(nn.Module):
    def __init__(self, n_class = 1000):
        super().__init__()

        self.backbone = iresnet50()

        self.backdoor_model = ConfounderNet(512, 7, 512, 2)
        self.surrogate_model = SurrogateNet(512, 7, 2, n_class)

        self.margin_loss = ArcMarginProduct(512, n_class)

        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()


    def set_requires_grad(self, nets: list[nn.Module], requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad



