import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from data import *
from Modules import *
from utils import *


def train_one_epoch(cfg, device, epoch, generator, discriminator, train_loader, optimizer_g, optimizer_d):
    f'''
    follows standard training procedure of Vanilla GAN / Wasserstein GAN
    k-steps for Discriminator (default : k=1)
    '''
    generator.train()
    discriminator.train()
    g_losses, d_losses = [], []
    pbar = tqdm(enumerate(train_loader), total=len(
        train_loader), position=0, leave=True)
    for _, (imgs, _) in pbar:
        if cfg.model_name == "DCGAN":
            Real = torch.ones(imgs.shape[0], requires_grad=False).to(device)
            Fake = torch.zeros(imgs.shape[0], requires_grad=False).to(device)
        imgs = imgs.to(device)
        # Train Discriminator
        for _ in range(cfg.discriminator_iteration):
            optimizer_d.zero_grad()
            z = torch.normal(0, 1, (imgs.shape[0], cfg.input_dim)).to(device)
            generated_imgs = generator(z)
            if cfg.model_name == "DCGAN":
                d_loss = get_criterion(cfg.model_name, imgs=imgs,
                                       real=Real, fake=Fake, generated_imgs=generated_imgs, discriminator=discriminator)
                d_losses.append(d_loss.item())
                d_loss.backward()
                optimizer_d.step()
            elif cfg.model_name == "WGAN":
                d_loss = get_criterion(cfg.model_name, imgs=imgs,
                                       generated_imgs=generated_imgs, discriminator=discriminator)
                d_losses.append(d_loss.item())
                d_loss.backward()
                optimizer_d.step()
                for p in discriminator.parameters():
                    p.data.clamp_(-cfg.clip_value, cfg.clip_value)
            elif cfg.model_name == "WGAN-GP":
                eps = torch.randn(imgs.shape[0], 1, 1, 1).to(device)
                d_loss = get_criterion(
                    cfg.model_name, imgs=imgs, generated_imgs=generated_imgs, discriminator=discriminator, eps=eps, device=device, lam=cfg.lam)
                d_losses.append(d_loss.item())
                d_loss.backward()
                optimizer_d.step()
                # Train Generator
        optimizer_g.zero_grad()
        z = torch.normal(0, 1, (imgs.shape[0], cfg.input_dim)).to(device)
        generated_imgs = generator(z)
        if cfg.model_name == "DCGAN":
            g_loss = get_criterion(cfg.model_name, imgs=None, real=Real, fake=Fake,
                                   generated_imgs=generated_imgs, discriminator=discriminator)
        elif cfg.model_name == "WGAN":
            # same generator loss for WGAN, WGAN-GP
            g_loss = get_criterion(
                cfg.model_name, imgs=None, generated_imgs=generated_imgs, discriminator=discriminator)
        elif cfg.model_name == "WGAN-GP":
            g_loss = get_criterion(
                cfg.model_name, imgs=None, generated_imgs=generated_imgs, discriminator=discriminator)
        g_losses.append(g_loss.item())
        g_loss.backward()
        optimizer_g.step()

        Train_description = f"Train Epoch : {epoch}, Discriminator Loss : {np.mean(d_losses): .6f}, Generator Loss : {np.mean(g_losses): .6f}"
        pbar.set_description(Train_description)
    return np.mean(g_losses)


def trainer(cfg, generator, discriminator, train_loader, val_loader, optimizer_g, optimizer_d):
    epoch = cfg.epoch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    generator.to(device)
    discriminator.to(device)
    if use_cuda:
        torch.cuda.empty_cache()
    best_loss = np.inf

    for e in range(epoch):
        generator_loss = train_one_epoch(cfg, device, e, generator, discriminator,
                                         train_loader, optimizer_g, optimizer_d)
        if best_loss > generator_loss:
            if not os.path.exists("./"+cfg.save_dir):
                os.mkdir("./"+cfg.save_dir)
            torch.save(generator.state_dict(), os.path.join(
                os.getcwd(), cfg.save_dir, f"{e}th.pt"))
            best_loss = generator_loss
            print("Best Model Changed")


def main(cfg):

    if cfg.model_name not in ["DCGAN", "WGAN", "WGAN-GP"]:
        raise RuntimeError(f"{cfg.model_name} is not implemented yet!!")

    seed_everything(cfg.seed)

    generator = Generator(cfg.input_dim, cfg.expanded_dim,
                          (cfg.img_size))
    discriminator = Discriminator()
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    optimizer_g, optimizer_d = get_optimizers(cfg, generator, discriminator)

    train_loader, val_loader = get_train_valid_loader(cfg)
    best_model = trainer(cfg, generator, discriminator,
                         train_loader, val_loader, optimizer_g, optimizer_d)
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
    gan_parser.add_argument("--lam", type=float, default=10.0)
    gan_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    gan_parser.add_argument("--epoch", type=int, default=100)
    gan_parser.add_argument("--input_dim", type=int, default=100)
    gan_parser.add_argument("--expanded_dim", type=int, default=128)
    gan_parser.add_argument("--lr", type=float, default=0.0002)
    gan_parser.add_argument("--discriminator_iteration", type=int, default=1)
    gan_parser.add_argument("--save_dir", type=str, default="models")
    cfg = gan_parser.parse_args()
    best_model = main(cfg)
