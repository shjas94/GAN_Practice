import torch


def get_optimizers(cfg, generator, discriminator):
    optimizer_g, optimizer_d = None, None
    if cfg.model_name == "DCGAN":
        optimizer_g = torch.optim.Adam(
            generator.parameters(), lr=cfg.lr, eps=1e-04, betas=[0.5, 0.999])
        optimizer_d = torch.optim.Adam(
            discriminator.parameters(), lr=cfg.lr, eps=1e-04, betas=[0.5, 0.999])
    elif cfg.model_name == "WGAN":
        optimizer_g = torch.optim.RMSprop(generator.parameters(), lr=cfg.lr)
        optimizer_d = torch.optim.RMSprop(
            discriminator.parameters(), lr=cfg.lr)
    elif cfg.model_name == "WGAN-GP":
        optimizer_g = torch.optim.Adam(
            generator.parameters(), lr=cfg.lr, eps=1e-04, betas=[0, 0.9])
        optimizer_d = torch.optim.Adam(discriminator.parameters(
        ), lr=cfg.lr, eps=1e-04, betas=[0, 0.9], weight_decay=1e-03)
    return optimizer_g, optimizer_d
