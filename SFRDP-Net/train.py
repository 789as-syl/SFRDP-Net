from model import Model
from option import *
from data_utils import get_dataloader
from tensorboardX import SummaryWriter
from Net import Net as Net
from math import ceil
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_loader,test_loader = get_dataloader(opt)
model = Model(Net,opt)
model.print_model()
writer = SummaryWriter(log_dir=opt.logdir)
for epoch in range(opt.total_epoch):
    model.scheduler.step()
    lr = model.scheduler.get_last_lr()[0]
    loss = model.optimize_parameters(train_loader,epoch)
    with torch.no_grad():
        psnr,ssim = model.test(test_loader)
        print("psnr:",psnr,"ssim:",ssim)
        writer.add_scalar('lr', lr, epoch)
        writer.add_scalar('psnr', psnr, epoch)
        writer.add_scalar('ssim', ssim, epoch)
        writer.add_scalar('train_loss', loss, epoch)
        model.save_network(epoch,psnr,ssim)
writer.close()

