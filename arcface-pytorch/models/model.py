import torch 
import torch.nn as nn
import torch.nn.functional as F
from .iresnet import iresnet50, IResNet
from .metrics import ArcMarginProduct
from .neural_backdoor import ConfounderNet, SurrogateNet

from torch.optim import AdamW

class DeconfoundedModel(nn.Module):
    def __init__(self, n_class = 1000, n_confounder = 2):
        super().__init__()

        self.n_class = n_class
        self.n_confounder = n_confounder

        self.backbone = iresnet50()

        self.set_requires_grad(self.backbone, False)
        self.backbone.eval()

        self.backdoor_model = ConfounderNet(512, 7, 512, n_confounder)
        self.surrogate_model = SurrogateNet(512, 7, 2, n_class)

        self.margin_loss = ArcMarginProduct(512, n_class)

        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()


        dis_params = [{'params': self.surrogate_model.invariant_feature_discriminator.parameters()},
                        {'params':self.surrogate_model.confounder_discriminator.parameters()}]
        self.optimizer_Dis = torch.optim.AdamW(dis_params, lr= 2e-4, betas=(0.9, 0.999), weight_decay=0.0001)


        feat_params = [{'params':self.backdoor_model.invariant_feature_generator.parameters()},
                        {'params':self.surrogate_model.confounder_generator.parameters()},
                        {'params':self.backdoor_model.confounders},
                        {'params':self.backdoor_model.feature_reconstructor.parameters()}]
        self.optimizer_Feat = torch.optim.AdamW(feat_params, lr=2e-4, betas=(0.9, 0.999), weight_decay=0.0001)        


        cls_params = [
                        # {'params': self.netBase.parameters(), 'lr': 1e-5},
                        {'params': self.backdoor_model.projector.parameters()},
                        {'params':self.margin_loss.parameters()}]
        self.optimizer_Cls = torch.optim.AdamW(cls_params, lr=2e-4, betas=(0.9, 0.999), weight_decay=0.0001)


    def train_step(self, data: list[torch.Tensor, torch.Tensor, torch.Tensor]):
        x, labels, confound_labels = data 
        
        ### Forward features
        features = self.backbone(x)
        invariant_features: torch.Tensor = self.backdoor_model.get_invariant_features(features)
        confounder_features: torch.Tensor = self.surrogate_model.get_confounder_features(invariant_features)

        reconstructed_features = self.backdoor_model.get_reconstruction_features(invariant_features, confounder_features)


        ### Training discriminators
        self.set_requires_grad(self.surrogate_model.confounder_discriminator, True)
        self.set_requires_grad(self.surrogate_model.invariant_feature_discriminator, True)
        invariant_features_dis = self.surrogate_model.invariant_feature_discriminator(invariant_features.detach())
        confounder_features_dis = self.surrogate_model.confounder_discriminator(confounder_features.detach())    

        invariant_dis_loss = self.ce_loss(invariant_features_dis, labels)
        confounder_dis_loss = self.ce_loss(confounder_features_dis, confound_labels)
        dis_loss: torch.Tensor = (invariant_dis_loss + confounder_dis_loss)/2
        self.optimizer_Dis.zero_grad()
        dis_loss.backward()
        self.optimizer_Dis.step()


        ### Training features
        self.set_requires_grad(self.surrogate_model.confounder_discriminator, False)
        self.set_requires_grad(self.surrogate_model.invariant_feature_discriminator, False)
        invariant_confuse_dis = self.surrogate_model.invariant_feature_discriminator(confounder_features) 
        confounder_confuse_dis = self.surrogate_model.confounder_discriminator(invariant_features)

        invariant_confuse_loss = self.rce_loss(invariant_confuse_dis, torch.ones((x.shape[0], self.n_class), device=labels.device) / self.n_class)
        confounder_confuse_loss = self.rce_loss(confounder_confuse_dis, torch.ones((x.shape[0], self.n_confounder), device=confound_labels.device) / self.n_confounder)

        loss_gen = (invariant_confuse_loss + confounder_confuse_loss) / 2
        loss_rec = self.mse_loss(reconstructed_features, invariant_features)
        loss_center = self.backdoor_model.calculate_center_loss(confounder_features, confound_labels) * 5e-4
        loss: torch.Tensor = loss_gen + loss_rec + loss_center

        self.optimizer_Feat.zero_grad() 
        loss.backward(retain_graph=True) 

        self.backdoor_model.confounders.grad.data =  self.backdoor_model.confounders.grad.data * (1. / 5e-4)
        self.optimizer_Feat.step()

        ### Training classifier
        intervened_features = self.backdoor_model.get_intervened_features(invariant_features)
        projected_features = self.backdoor_model.projector(intervened_features)
        pred_logits: torch.Tensor = self.margin_loss(projected_features, labels)
        pred_logits = pred_logits.view(-1, self.n_confounder, self.margin_loss.out_features).mean(dim=1)

        cls_loss: torch.Tensor = self.ce_loss(pred_logits, labels)
        self.optimizer_Cls.zero_grad()
        cls_loss.backward()
        self.optimizer_Cls.step()

        accuracy = (torch.argmax(pred_logits, dim=1) == labels).float().mean()

        return {
            'dis_loss': dis_loss.item(),
            'feat_loss': loss.item(),
            'cls_loss': cls_loss.item(),
            'loss_rec': loss_rec.item(),
            'loss_center': loss_center.item(),
            'loss_gen': loss_gen.item(),
            'accuracy': accuracy.item(),
        }

    def test_step(self, data: list[torch.Tensor, torch.Tensor]):
        x, labels = data
        features = self.backbone(x)
        invariant_features: torch.Tensor = self.backdoor_model.get_invariant_features(features)
        intervened_features = self.backdoor_model.get_intervened_features(invariant_features)
        projected_features = self.backdoor_model.projector(intervened_features)
        pred_logits: torch.Tensor = self.margin_loss.test(projected_features)
        pred_logits = pred_logits.view(-1, self.n_confounder, self.margin_loss.out_features).mean(dim=1)

        cls_loss: torch.Tensor = self.ce_loss(pred_logits, labels)

        accuracy = (torch.argmax(pred_logits, dim=1) == labels).float().mean()

        return {
            'cls_loss': cls_loss.item(),
            'accuracy': accuracy.item(),
        }


    def rce_loss(self, x, y):
        # RCE
        pred, labels = x, y

        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
        # Loss
        loss = rce.mean()        
        return loss

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



