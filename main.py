import os
import argparse
from torch.optim import optimizer
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from data import *
from Modules import *
from utils import *


def train_one_epoch(cfg, device, epoch, generator, discriminator, train_loader, criterion, optimizer_g, optimizer_d):
    f'''
    follows standard training procedure of Vanilla GAN / Wasserstein GAN
    k-steps for Discriminator k=1 for the implementation

    '''
    generator.train()
    discriminator.train()

    pbar = tqdm(enumerate(train_loader), total=len(
        train_loader), position=0, leave=True)
    for iter, (imgs, _) in pbar:
        Real = torch.ones(imgs.shape[0], requires_grad=False).to(device)
        Fake = torch.zeros(imgs.shape[0], requires_grad=False).to(device)
        imgs = imgs.to(device)

        # Train Discriminator
        for d_iter in range(cfg.discriminator_iteration):
            z = torch.normal(0, 1, (imgs.shape[0], cfg.input_dim)).to(device)
            generated_imgs = generator(z)
            optimizer_d.zero_grad()
            if cfg.model_name == "DCGAN":
                real_loss = criterion(discriminator(imgs), Real)
                fake_loss = criterion(discriminator(
                    generated_imgs.detach()), Fake)
                d_loss = real_loss + fake_loss
                d_loss.backward()
                optimizer_d.step()
            elif cfg.model_name == "WGAN":
                d_loss = -torch.mean(discriminator(imgs)) + \
                    torch.mean(discriminator(generated_imgs.detach()))
                d_loss.backward()
                optimizer_d.step()
                for p in discriminator.parameters():
                    p.data.clamp_(-cfg.clip_value, cfg.clip_value)

        # Train Generator
        z = torch.normal(0, 1, (imgs.shape[0], cfg.input_dim)).to(device)
        generated_imgs = generator(z)
        optimizer_g.zero_grad()
        if cfg.model_name == "DCGAN":
            g_loss = criterion(discriminator(generated_imgs), Real)
        elif cfg.model_name == "WGAN":
            g_loss = -torch.mean(discriminator(generated_imgs))
        g_loss.backward()
        optimizer_g.step()

        Train_description = f"Train Epoch : {epoch}, Discriminator Loss : {d_loss.item(): .6f}, Generator Loss : {g_loss.item(): .6f}"
        pbar.set_description(Train_description)


def valid_one_epoch(cfg, device, epoch, generator, discriminator, val_loader, criterion):
    generator.eval()
    discriminator.eval()

    pbar = tqdm(enumerate(val_loader), total=len(
        val_loader), position=0, leave=True)
    g_loss = None
    for iter, (imgs, _) in pbar:
        imgs = imgs.to(device)
        z = torch.normal(0, 1, (imgs.shape[0], cfg.input_dim)).to(device)
        generated_imgs = generator(z)
        if cfg.model_name == "DCGAN":
            Real = torch.ones(imgs.shape[0], requires_grad=False).to(device)
            Fake = torch.zeros(imgs.shape[0], requires_grad=False).to(device)
            real_loss = criterion(discriminator(imgs), Real)
            fake_loss = criterion(discriminator(generated_imgs.detach()), Fake)
            d_loss = real_loss + fake_loss
            g_loss = criterion(discriminator(generated_imgs), Real)

        elif cfg.model_name == "WGAN":
            d_loss = -torch.mean(discriminator(imgs)) + \
                torch.mean(discriminator(generated_imgs))
            g_loss = -torch.mean(discriminator(generated_imgs))

        valid_description = f"Valid Epoch : {epoch}, Discriminator Loss : {d_loss.item(): .6f}, Generator Loss : {g_loss.item(): .6f}"
        pbar.set_description(valid_description)
    return g_loss.item()


def trainer(cfg, generator, discriminator, train_loader, val_loader, criterion, optimizer_g, optimizer_d):
    epoch = cfg.epoch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    generator.to(device)
    discriminator.to(device)
    if use_cuda:
        torch.cuda.empty_cache()
    best_loss = np.inf

    for e in range(epoch):
        train_one_epoch(cfg, device, e, generator, discriminator,
                        train_loader, criterion, optimizer_g, optimizer_d)
        with torch.no_grad():
            generator_loss = valid_one_epoch(
                cfg, device, e, generator, discriminator, val_loader, criterion)
        if best_loss > generator_loss:
            if not os.path.exists("./"+cfg.save_dir):
                os.mkdir("./"+cfg.save_dir)
            torch.save(generator.state_dict(), os.path.join(
                os.getcwd(), cfg.save_dir, f"{e}th.pt"))
            best_loss = generator_loss
            print("Best Model Changed")


def main(cfg):
    seed_everything(cfg.seed)

    if cfg.dataset_name == "cifar-10":
        num_classes = 10
    elif cfg.dataset_name == "cifar-100":
        num_classes = 100
    elif cfg.dataset_name == "imagenet":
        num_classes = 1000
    else:
        print("need to implement custom dataset/loader")
        return
    criterion = nn.BCELoss()
    generator = Generator(cfg.input_dim, cfg.expanded_dim,
                          (cfg.img_size))
    discriminator = Discriminator(num_classes)
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    if cfg.model_name == "DCGAN":
        optimizer_g = torch.optim.Adam(
            generator.parameters(), lr=cfg.lr, eps=1e-04, betas=[0.5, 0.999])
        optimizer_d = torch.optim.Adam(
            discriminator.parameters(), lr=cfg.lr, eps=1e-04, betas=[0.5, 0.999])
    elif cfg.model_name == "WGAN":
        optimizer_g = torch.optim.RMSprop(generator.parameters(), lr=cfg.lr)
        optimizer_d = torch.optim.RMSprop(
            discriminator.parameters(), lr=cfg.lr)

    train_loader, val_loader = get_train_valid_loader(cfg)
    best_model = trainer(cfg, generator, discriminator,
                         train_loader, val_loader, criterion, optimizer_g, optimizer_d)
    return best_model


if __name__ == "__main__":
    gan_parser = argparse.ArgumentParser()
    gan_parser.add_argument("--model_name", type=str, default="DCGAN",
                            help="Name of GAN Architecture\n(DCGAN\nWGAN\nWGAN-GP)")
    gan_parser.add_argument("--dataset_name", type=str,
                            default="cifar-10", help="Name of Dataset")
    gan_parser.add_argument("--train_size", type=float,
                            default=0.8, help="Ratio of Training set")
    gan_parser.add_argument("--img_size", type=int,
                            default=64, help="Size of input image")
    gan_parser.add_argument("--batch_size", type=int,
                            default=16, help="size of train batches")
    gan_parser.add_argument("--test_batch_size", type=int,
                            default=16, help="size of test batches")
    gan_parser.add_argument("--clip_value", type=float, default=0.01,
                            help="values for clipping parameters of Discriminator to enforce a\
                            Lipschitz constant (requires only for WGAN)")
    gan_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    gan_parser.add_argument("--epoch", type=int, default=100)
    gan_parser.add_argument("--input_dim", type=int, default=100)
    gan_parser.add_argument("--expanded_dim", type=int, default=128)
    gan_parser.add_argument("--lr", type=float, default=0.0002)
    gan_parser.add_argument("--discriminator_iteration", type=int, default=1)
    gan_parser.add_argument("--save_dir", type=str, default="models")
    cfg = gan_parser.parse_args()

    best_model = main(cfg)
