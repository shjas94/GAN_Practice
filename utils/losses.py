import torch
import torch.nn as nn


def adversarial_loss(imgs, real, fake, generated_imgs, discriminator):
    bce = nn.BCELoss()
    if imgs is not None:
        return bce(discriminator(imgs).squeeze(), real) + bce(discriminator(generated_imgs).squeeze(), fake)
    else:
        return bce(discriminator(generated_imgs).squeeze(), real)


def wasserstein_loss(imgs, generated_imgs, discriminator):
    if imgs is not None:
        return -torch.mean(discriminator(imgs)) + torch.mean(discriminator(generated_imgs))
    else:
        return -torch.mean(discriminator(generated_imgs))


def gradient_penalty(imgs, generated_imgs, discriminator, eps, device):
    x_hat = (imgs*eps + generated_imgs*(1-eps)).requires_grad_(True)
    d_x_hat = discriminator(x_hat)
    fake = torch.ones((d_x_hat.shape[0], 1), requires_grad=False).to(device)
    grad = torch.autograd.grad(outputs=d_x_hat, inputs=x_hat, grad_outputs=fake,
                               create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad = grad.view(grad.shape[0], -1)
    return torch.mean((torch.linalg.norm(grad) - 1) ** 2)


def wasserstein_gp_loss(imgs, generated_imgs, discriminator, eps=None, device=None, lam=None):
    if imgs is not None:
        return -torch.mean(discriminator(imgs)) + torch.mean(discriminator(generated_imgs)) + lam * gradient_penalty(
            imgs, generated_imgs, discriminator, eps, device)
    else:
        return -torch.mean(discriminator(generated_imgs))


_criterion_entrypoints = {
    'DCGAN': adversarial_loss,
    'WGAN': wasserstein_loss,
    'WGAN-GP': wasserstein_gp_loss
}


def criterion_entrypoints(criterion_name):
    return _criterion_entrypoints[criterion_name]


def is_criterion(model_name):
    return model_name in _criterion_entrypoints


def get_criterion(model_name, **kwargs):
    if is_criterion(model_name):
        create_fn = criterion_entrypoints(model_name)
        criterion = create_fn(**kwargs)
    else:
        raise RuntimeError(f"Unknown Model Name {model_name}")
    return criterion
