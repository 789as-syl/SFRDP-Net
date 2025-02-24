from tqdm import tqdm
from torch.backends import cudnn
from torch import optim
from pytorch_msssim import *
from metrics import *
import torch.nn as nn
import torch
import numpy as np
import torchvision.utils as vutils
import os

class Model(nn.Module):
    def __init__(self, Net, opts):
        super().__init__()
        self.opt = opts
        self.start_epoch = 0
        self.best_psnr = 0
        self.best_ssim = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.scaler = torch.cuda.amp.GradScaler()
        self.model = Net().to(self.device)
        self.optimizer = optim.AdamW(params=filter(lambda x: x.requires_grad, self.model.parameters()), lr=opts.lr,
                                     betas=(0.9, 0.999),
                                     eps=1e-08, amsgrad=False, weight_decay=0.01)

        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.opt.total_epoch,
                                                              eta_min=self.opt.lr * 0.05)
        self.set_seed(opts.seed)

    def set_seed(self, seed):
        seed = int(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    def save_network(self, epoch, now_psnr, now_ssim):
        if self.best_psnr < now_psnr and self.best_ssim < now_ssim:
            self.best_psnr = now_psnr
            self.best_ssim = now_ssim
            model_path = os.path.join(self.opt.model_Savepath, 'best_model.pth')
            opt_path = os.path.join(self.opt.optim_Savepath, 'best_opt.pth')
        elif epoch % self.opt.save_fre_step == 0:
            model_path = os.path.join(self.opt.model_Savepath, 'E{}_model.pth'.format(epoch))
            opt_path = os.path.join(self.opt.optim_Savepath, 'E{}_opt.pth'.format(epoch))
        else:
            return

        # model_save
        torch.save(
            self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict(),
            model_path)
        # optim_save
        opt_state = {'epoch': epoch, 'ssim': now_ssim, 'psnr': now_psnr,
                     'scheduler': self.scheduler.state_dict(), 'optimizer': self.optimizer.state_dict()}
        torch.save(opt_state, opt_path)

    def load_network(self):
        model_path = self.opt.model_loadPath
        opt_path = self.opt.opt_loadPath
        self.model.load_state_dict(torch.load(model_path))
        optim_state = torch.load(opt_path)
        self.start_epoch = optim_state['epoch']
        self.best_psnr = optim_state['psnr']
        self.best_ssim = optim_state['ssim']
        self.optimizer.load_state_dict(optim_state['optimizer'])
        self.scheduler.load_state_dict(optim_state['scheduler'])
        print(self.best_psnr)
        print(self.best_ssim)

    def optimize_parameters(self, train_dataloader, epoch):
        self.model.train()
        total_loss = 0.0
        for idx, (input_img, label_img) in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch}', leave=True)):
            input_img = input_img.to(self.device)
            label_img = label_img.to(self.device)
            loss = self.model(label_img,input_img)
            loss.backward()
            total_loss += loss.item()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        return total_loss / len(train_dataloader)

    def test(self, test_dataloder):
        self.model.eval()
        # torch.cuda.empty_cache()
        ssims = []
        psnrs = []
        for step, (inputs, targets) in enumerate(test_dataloder):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            pred = self.model(inputs)
            ssims.append(ssim(pred, targets).item())
            psnrs.append(psnr(pred, targets))
        ssim_mean = np.mean(ssims)
        psnr_mean = np.mean(psnrs)
        return psnr_mean, ssim_mean

    def print_model(self):
        pytorch_total_params = sum(p.nelement() for p in self.model.parameters() if p.requires_grad)
        print("Total_params: ==> {}".format(pytorch_total_params / 1e6))


