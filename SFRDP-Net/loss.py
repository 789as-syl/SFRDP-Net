import torch.nn as nn
import torch

from torchvision import models

class fftLoss(nn.Module):
    def __init__(self):
        super(fftLoss, self).__init__()

    def forward(self, x, y):
        diff = torch.fft.fft2(x.to('cuda:0')) - torch.fft.fft2(y.to('cuda:0'))
        loss = torch.mean(abs(diff))
        return loss

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

class ContrastLoss(nn.Module):
    def __init__(self):
        super(ContrastLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, a, p):
        a_vgg, p_vgg = self.vgg(a), self.vgg(p)
        loss = 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            contrastive = d_ap
            loss += self.weights[i] * contrastive
        return loss


class PFDC(nn.Module):
    def __init__(self):
        super(PFDC, self).__init__()
        self.vgg = Vgg19().cuda()
        self.smooth_l1 = nn.SmoothL1Loss().cuda()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def loss_formulation(self, x, y):
        B, C, H, W = x.shape
        x_mean = x.mean(dim=[2,3],keepdim=True)
        y_mean = y.mean(dim=[2, 3], keepdim=True)
        dis = torch.abs(x_mean-y_mean)
        dis_max =  torch.max(dis,dim=1)[0].view(B,1,1,1)
        dis = dis / dis_max
        dis = torch.exp(dis / 1.0)-0.3
        dis = dis.detach()
        return dis

    def forward(self, out, y):
        out_vgg, y_vgg = self.vgg(out), self.vgg(y)
        loss = 0
        for i in range(len(out_vgg)):
            w = self.loss_formulation(out_vgg[i], y_vgg[i].detach())
            contrastive = self.smooth_l1(out_vgg[i] * w, y_vgg[i].detach() * w)
            loss += self.weights[i] * contrastive
        return loss
