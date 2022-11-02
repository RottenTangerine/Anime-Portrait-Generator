import time
import os

import torch
from config import get_parser
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable
from dataset.dataset import ImageData

from model import discriminator, generator
from model.utils import init_net

from torchvision import transforms
from icecream import ic

args = get_parser()
train_id = int(time.time())
resume_epoch = 0
print(f'Training ID: {train_id}')

dataset = ImageData(args.dataset)
data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

print(f'Train data batches: {len(data_loader)}')

device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'

# model
G = init_net(generator.Generator(args.noise_features), args.init_type, args.init_gain).to(device)
D = init_net(discriminator.Discriminator(args.channel), args.init_type, args.init_gain).to(device)
print(G.eval())

# retrained / continuous training
try:
    most_recent_check_point = os.listdir('checkpoint')[-1]
    ckpt_path = os.path.join('checkpoint', most_recent_check_point)
    check_point = torch.load(ckpt_path)
    # load model
    G.load_state_dict(check_point['G_state_dict'])
    D.load_state_dict(check_point['D_state_dict'])
    resume_epoch = check_point['epoch']
    print(f'Successfully load checkpoint {most_recent_check_point}, '
          f'start training from epoch {resume_epoch + 1}')
except:
    print('fail to load checkpoint, train from zero beginning')

# optimizer and schedular
optimizer_G = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

lr_scheduler_G = lr_scheduler.CosineAnnealingWarmRestarts(optimizer_G, T_0=10, T_mult=2, eta_min=1e-5)
lr_scheduler_D = lr_scheduler.ExponentialLR(optimizer_D, gamma=0.9)

# criterion
criterion_GAN = torch.nn.MSELoss()

print('***start training***')
# training
for epoch in range(resume_epoch + 1, args.epochs):
    epoch_start_time = time.time()
    print(f'{"*" * 20} Start epoch {epoch}/{args.epochs} {"*" * 20}')

    for i, real_img in enumerate(data_loader):
        real_img = real_img.to(device)

        z = torch.randn((real_img.shape[0], args.noise_features, 1, 1)).to(device)
        fake_img = G(z)

        # Generator
        optimizer_G.zero_grad()

        pred_fake = D(fake_img)
        target_real = (torch.ones(pred_fake.shape) * 0.95).to(device)
        target_fake = (torch.ones(pred_fake.shape) * 0.05).to(device)
        loss_G = criterion_GAN(pred_fake, target_real)

        loss_G.backward()
        optimizer_G.step()

        # Discriminator
        optimizer_D.zero_grad()

        pred_real = D(real_img)
        loss_pred_real = criterion_GAN(pred_real, target_real)
        # ic(pred_real, loss_pred_real)
        pred_fake = D(fake_img.detach())
        loss_pred_fake = criterion_GAN(pred_fake, target_fake)
        # ic(pred_fake, loss_pred_fake)
        loss_D = (loss_pred_real + loss_pred_fake) * 0.5

        loss_D.backward()
        optimizer_D.step()

        if i % args.print_interval == 0:
            print(f'epoch: {epoch}/{args.epochs}\tbatch: {i}/{len(data_loader)}\t'
                  f'loss_G: {loss_G:0.6f}\tloss_D: {loss_D:0.6f}\t'
                  f'|| learning rate_G: {optimizer_G.state_dict()["param_groups"][0]["lr"]:0.6f}\t'
                  f'learning rate_D: {optimizer_D.state_dict()["param_groups"][0]["lr"]:0.8f}\t')

            os.makedirs('output', exist_ok=True)
            trans = transforms.ToPILImage()
            trans(fake_img[0]).save(f'output/{train_id}_{epoch}_{i}_fake.jpg')

    # scheduler
    lr_scheduler_G.step()
    lr_scheduler_D.step()

    print(f'End of epoch: {epoch}/{args.epochs}\t time taken: {time.time() - epoch_start_time:0.2f}')

    # save ckpt
    os.makedirs('checkpoint', exist_ok=True)
    torch.save({'epoch': epoch,
                'G_state_dict': G.state_dict(),
                'D_state_dict': D.state_dict(),
                }, f'checkpoint/{train_id}_{epoch:03d}.pt')

# save model
os.makedirs('trained_model', exist_ok=True)
torch.save(G.state_dict(), f'./trained_model/G_B2A_{train_id}.pth')
