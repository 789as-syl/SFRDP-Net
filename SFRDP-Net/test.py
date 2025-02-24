from model import Model
from option import *
from data_utils import get_dataloader
from tensorboardX import SummaryWriter
from Net import Net
from PIL import Image
from torchvision import transforms
import torchvision.utils as vutils
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(opt.logdir)
train_loader,test_loader = get_dataloader(opt)
model = Model(Net,opt)
model.load_network()
with torch.no_grad():
    psnr,ssim = model.test(test_loader)
    print("psnr:",psnr,"ssim:",ssim)

transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量
])

image_path = "/T2020027/yzr/data/Hazy_DetRDDTS/images/00495.jpg"
input_image = Image.open(image_path).convert('RGB')
input_tensor = transform(input_image).unsqueeze(0).to(device)
# 模型推理
model = model.model
model.load_state_dict(torch.load("/T2020027/yzr/projects/SFNet/train_models/SFNet_DHID/E32_model.pth"))
model.eval()  # 切换到评估模式
with torch.no_grad():
    output_tensor = model(input_tensor)
output_path = "RDDTF.jpg"  # 指定输出路径
vutils.save_image(output_tensor.cpu(), output_path)

