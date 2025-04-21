import torch


def fgsm_penalty(source_tensor, target_latent, modifier, latent_function, eps=0.05):

    # Implementation of the Fast Gradient Sign Method (FGSM) for perturbation
    # as seen in https://arxiv.org/pdf/1607.02533

    # Ensure the modifier and source_tensor are set to require gradients
    modifier = modifier.clone().requires_grad_(True)
    source_tensor = source_tensor.clone().requires_grad_(True)


    adv_tensor = torch.clamp(modifier + source_tensor, -1, 1)
    adv_latent = latent_function(adv_tensor)
    loss = (adv_latent - target_latent).norm()

    grad = torch.autograd.grad(loss.sum(), modifier, retain_graph=True)[0]

    perturbation = eps * torch.sign(grad)
    adv_tensor = torch.clamp(source_tensor + perturbation, -1, 1)
    return adv_tensor - source_tensor

def pgd_penalty(source_tensor, target_latent, modifier, latent_function, iterations=10, step_size=0.01, eps=0.05):

    # Implementation of the Projected Gradient Descent (PGD) method for perturbation
    # as seen in https://arxiv.org/pdf/1706.06083

    # Ensure the modifier and source_tensor are set to require gradients
    modifier = modifier.clone().requires_grad_(True)
    source_tensor = source_tensor.clone().requires_grad_(True)


    perturbation = torch.zeros_like(modifier)

    # Iteratively update the perturbation
    for _ in range(iterations):
        adv_tensor = torch.clamp(modifier + perturbation + source_tensor, -1, 1)
        adv_latent = latent_function(adv_tensor)
        loss = (adv_latent - target_latent).norm()

        grad = torch.autograd.grad(loss.sum(), modifier, retain_graph=True)[0]
        perturbation = perturbation + step_size * torch.sign(grad)
        perturbation = torch.clamp(perturbation, -eps, eps)
    
    return perturbation

def nightshade_penalty(source_tensor, target_latent, modifier, latent_function, t_size=100, eps=0.05, max_change=0):

    # Implementation of the Nightshade method for perturbation
    # as seen in https://arxiv.org/pdf/2310.13828
    max_change = eps / 0.5
    step_size = max_change

    # Iteratively update the modifier
    for i in range(t_size):
        actual_step_size = step_size - (step_size - step_size/100)/t_size * i
        modifier.requires_grad_(True)

        adv_tensor = torch.clamp(modifier + source_tensor, -1, 1)
        adv_latent = latent_function(adv_tensor)

        loss = (adv_latent - target_latent).norm()

        grad = torch.autograd.grad(loss.sum(), modifier)[0]
        modifier = modifier - torch.sign(grad) * actual_step_size
        modifier = torch.clamp(modifier, -max_change, max_change).detach()
    
    return modifier
