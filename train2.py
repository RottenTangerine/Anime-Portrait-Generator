import time
import os

import torch
import torch.nn as nn

from config import get_parser
from torch.utils.data import DataLoader
from dataset.dataset import ImageData
from model.utils import init_net
from generate import gen_img
from model import discriminator, generator
import torchvision


args = get_parser()
train_id = int(time.time())
resume_epoch = 0
print(f'Training ID: {train_id}')

dataset = ImageData(args.dataset)
data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

print(f'Train data batches: {len(data_loader)}')

device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'

# model
discriminator = nn.Sequential(
    # in: 3 x 64 x 64

    nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 64 x 32 x 32

    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 128 x 16 x 16

    nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 256 x 8 x 8

    nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 512 x 4 x 4

    nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
    # out: 1 x 1 x 1

    nn.Flatten(),
    nn.Sigmoid())


generator = nn.Sequential(
    # in: latent_size x 1 x 1

    nn.ConvTranspose2d(args.noise_features, 512, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    # out: 512 x 4 x 4
    nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    # out: 512 x 4 x 4

    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    # out: 256 x 8 x 8

    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    # out: 128 x 16 x 16

    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    # out: 64 x 32 x 32

    nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
    # out: 3 x 64 x 64
)

# G = init_net(generator.Generator(args.noise_features), args.init_type, args.init_gain).to(device)
# D = init_net(discriminator.Discriminator(args.channel), args.init_type, args.init_gain).to(device)

G = init_net(generator).to(device)
D = init_net(discriminator).to(device)

# optimizer and schedular
optimizer_G = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

criterion = nn.functional.binary_cross_entropy


def train_discriminator(real_images, opt_d):
    # Clear discriminator gradients
    opt_d.zero_grad()

    # Pass real images through discriminator
    real_preds = D(real_images)
    real_targets = torch.ones(real_images.size(0), 1, device=device)
    real_loss = criterion(real_preds, real_targets)
    real_score = torch.mean(real_preds).item()

    # Generate fake images
    latent = torch.randn(args.batch_size, args.noise_features, 1, 1, device=device)
    fake_images = G(latent)

    # Pass fake images through discriminator
    fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
    fake_preds = D(fake_images)
    fake_loss = criterion(fake_preds, fake_targets)
    fake_score = torch.mean(fake_preds).item()

    # Update discriminator weights
    loss = real_loss + fake_loss
    loss.backward()
    opt_d.step()
    return loss.item(), real_score, fake_score


def train_generator(opt_g):
    # Clear generator gradients
    opt_g.zero_grad()

    # Generate fake images
    latent = torch.randn(args.batch_size, args.noise_features, 1, 1, device=device)
    fake_images = G(latent)

    # Try to fool the discriminator
    preds = D(fake_images)
    targets = torch.ones(args.batch_size, 1, device=device)
    loss = criterion(preds, targets)

    # Update generator weights
    loss.backward()
    opt_g.step()

    return loss.item()


print('***start training***')
# training
losses_g = []
losses_d = []
real_scores = []
fake_scores = []

# Create optimizers

for epoch in range(args.epochs):
    for i, real_images in enumerate(data_loader):
        real_images = real_images.to(device)
        loss_d, real_score, fake_score = train_discriminator(real_images, optimizer_D)
        loss_g = train_generator(optimizer_G)

        if i % args.print_interval == 0:
            print(f'epoch: {epoch}/{args.epochs}\tbatch: {i}/{len(data_loader)}\t'
                  f'loss_G: {loss_g:0.6f}\tloss_D: {loss_d:0.6f}\t'
                  f'|| learning rate_G: {optimizer_G.state_dict()["param_groups"][0]["lr"]:0.8f}\t'
                  f'learning rate_D: {optimizer_D.state_dict()["param_groups"][0]["lr"]:0.8f}\t')

            # ic(pred_fake, pred_real)
            # ic(pred_fake.shape)
            with torch.no_grad():
                os.makedirs('output', exist_ok=True)
                z = torch.randn((4, args.noise_features, 1, 1)).to(device)
                fake_img = G(z)
                img = torch.cat([torch.cat([gen_img(args, G, device) for _ in range(4)], dim=-1) for _ in range(4)], dim=-2)
                torchvision.utils.save_image(img, f'output/{train_id}_{epoch}_{i}.jpg')

    # Record losses & scores
    losses_g.append(loss_g)
    losses_d.append(loss_d)
    real_scores.append(real_score)
    fake_scores.append(fake_score)

# save model
os.makedirs('trained_model', exist_ok=True)
try:
    torch.save(G.state_dict(), f'./trained_model/G_{train_id}.pth')
    print(f'Successfully saves the model ./trained_model/G_{train_id}.pth')
except:
    print('Fail to save the model, this project will automatically use the latest checkpoint to recover the model')
