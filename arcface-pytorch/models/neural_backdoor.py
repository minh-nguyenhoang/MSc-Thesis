import torch
from torch import nn
from .initializer import init_weights
from .metrics import ArcMarginProduct

import functools

class ConfounderNet(nn.Module):
    """docstring for ConfounderNet"""

    def __init__(self, in_channels=512, feat_size=7, out_channels=512, confound_nc=2, ):
        super(ConfounderNet, self).__init__()

        self.confound_nc = confound_nc
        self.confounders = nn.Parameter(torch.zeros([confound_nc, in_channels, feat_size, feat_size]))

        self.register_buffer("classes", torch.arange(self.confound_nc).long())
        self.classes: torch.Tensor


        self.invariant_feature_generator = FeatEncoderNet(512, 2)
        self.feature_reconstructor = FeatDecoderNet(in_channels * 2, 3)

        
        self.projector = nn.Sequential(
            nn.Flatten(-3,-1),
            nn.Linear(in_channels*feat_size*feat_size, out_channels))

    # def forward(self, center_weights=None, is_test=False):
    #     if (not is_test) and (center_weights is not None):
    #         self.centers = center_weights.view(*list(self.centers.size()))

    #     return self.centers
    
    def get_intervened_features(self, x: torch.Tensor):
        bz = x.shape[0]
        x_shape = x.shape
        confounders = self.confounders.detach().unsqueeze(0).repeat_interleave(bz, 0).flatten(0,1)

        invariant_features: torch.Tensor = self.invariant_feature_generator(x)
        invariant_features = invariant_features.unsqueeze(1).repeat_interleave(self.confound_nc, 1).flatten(0,1)
        intervened_features = self.feature_reconstructor(invariant_features, confounders)
        # intervened_features = self.projector(intervened_features)
        
        return intervened_features
    
    def get_invariant_features(self, x):
        invariant_features: torch.Tensor = self.invariant_feature_generator(x)
        return invariant_features
    
    def get_reconstruction_features(self, x: torch.Tensor, confounders: torch.Tensor):
        invariant_features = x
        intervened_features = self.feature_reconstructor(invariant_features, confounders)
        return intervened_features

    
    def calculate_center_loss(self, x: torch.Tensor, y_cnfd: torch.Tensor):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim, feat_sz, feat_sz).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        labels = y_cnfd
        x = x.flatten(1,)
        centers = self.confounders.flatten(1,)

        x = x.view(batch_size, -1)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.confound_nc) + \
                  torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(self.confound_nc, batch_size).t()
        distmat.addmm_(1, -2, x, centers.t())

        # mask the corresponding class for each x
        # classes = torch.arange(self.num_classes).long()
        # if x.is_cuda:
        #     classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.confound_nc)
        mask = labels.eq(self.classes.clone().expand(batch_size, self.confound_nc))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss
    

class SurrogateNet(nn.Module):
    def __init__(self, in_channels=512, feat_size=7, confound_nc=2, n_class=1000):
        super(SurrogateNet, self).__init__()

        self.confounder_generator = FeatEncoderNet(in_channels, 2)

        self.confounder_discriminator = FeatDisNet(in_channels, confound_nc, feat_size, 3)
        self.invariant_feature_discriminator = FeatDisNet(in_channels, n_class, feat_size, 3)

    def get_confounder_features(self, x):
        confounder_features = self.confounder_generator(x)
        return confounder_features










class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class FeatEncoderNet(nn.Module):
    """docstring for FeatEncoderNet"""

    def __init__(self, in_channels, n_blocks=2, norm_layer=nn.BatchNorm2d):
        super(FeatEncoderNet, self).__init__()

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        layers = [nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)]
        for idx in range(n_blocks):
            layers += [ResnetBlock(in_channels, 'zero', norm_layer, False, use_bias)]

        self.feat_encoder = nn.Sequential(*layers)

    def forward(self, x):
        out_feat = self.feat_encoder(x)

        return out_feat


class FeatDecoderNet(nn.Module):
    """docstring for FeatDecoderNet"""

    def __init__(self, in_channels, n_blocks=3, norm_layer=nn.BatchNorm2d):
        super(FeatDecoderNet, self).__init__()

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        out_channels = int(in_channels / 2)
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
        for idx in range(n_blocks):
            layers += [ResnetBlock(out_channels, 'zero', norm_layer, False, use_bias)]

        self.feat_decoder = nn.Sequential(*layers)

    def forward(self, x1, x2):
        combined_x = torch.cat((x1, x2), 1)
        out_feat = self.feat_decoder(combined_x)

        return out_feat


class FeatDisNet(nn.Module):
    """docstring for FeatDisNet"""

    def __init__(self, in_channels=512, n_emo=6, feat_size=7, repeat_num=3):
        super(FeatDisNet, self).__init__()
        layers = []

        layers += [nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.01)]

        curr_dim = in_channels
        for i in range(repeat_num):
            layers += [nn.Conv2d(int(curr_dim), int(curr_dim / 2), kernel_size=1, stride=1, padding=0),
                        nn.LeakyReLU(0.01)]
            curr_dim = int(curr_dim / 2)

        kernel_size = int(feat_size / 2)
        layers += [nn.Conv2d(curr_dim, n_emo, kernel_size=kernel_size, bias=False)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        pred_expr = self.model(x)
        pred_expr = torch.squeeze(pred_expr, (-2,-1))
        return pred_expr