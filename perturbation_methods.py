import torch
import lpips
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lpips_loss = lpips.LPIPS(net='vgg').to(device)

def fgsm_penalty(source_tensor, target_latent, modifier, latent_function, eps=0.2):
    modifier = modifier.detach().requires_grad_(True)
    source_tensor = source_tensor.detach().requires_grad_(True)
    target_latent = target_latent.detach()

    adv_tensor = torch.clamp(modifier + source_tensor, -1, 1)
    adv_latent = latent_function(adv_tensor)
    loss = (adv_latent - target_latent).norm()

    grad = torch.autograd.grad(loss.sum(), modifier, retain_graph=False)[0]
    perturbation = eps * torch.sign(grad)
    adv_tensor = torch.clamp(source_tensor + perturbation, -1, 1)
    return (adv_tensor - source_tensor)

def pgd_penalty(source_tensor, target_latent, modifier, latent_function, iterations=10, step_size=0.05, eps=0.1):
    modifier = modifier.detach().requires_grad_(True)
    source_tensor = source_tensor.detach().requires_grad_(True)
    target_latent = target_latent.detach()
    
    perturbation = torch.zeros_like(modifier)

    for _ in range(iterations):
        adv_tensor = torch.clamp(modifier + perturbation + source_tensor, -1, 1)
        adv_latent = latent_function(adv_tensor)
        loss = (adv_latent - target_latent).norm()

        grad = torch.autograd.grad(loss.sum(), modifier, retain_graph=False)[0]
        perturbation = (perturbation + step_size * torch.sign(grad))
        perturbation = torch.clamp(perturbation, -eps, eps)
    
    return perturbation

def nightshade_penalty(
    source_tensor,
    target_latent,
    modifier,
    latent_function,
    t_size=20,
    eps=0.1,
    lpips_threshold=0.07,
    verbose=True,
):
    # Constants
    max_change = eps / 0.5
    step_size = max_change

    # Make sure all tensors are detached and clean
    source_tensor = source_tensor.detach()
    target_latent = target_latent.detach()
    modifier = modifier.detach()

    for i in range(t_size):
        actual_step_size = step_size - (step_size - step_size / 100) * (i / t_size)
        modifier.requires_grad_(True)
        adv_tensor = torch.clamp(modifier + source_tensor, -1.0, 1.0)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            adv_latent = latent_function(adv_tensor)

        # Compute MSE loss in latent space
        latent_loss = torch.nn.functional.mse_loss(adv_latent, target_latent)

        # Backpropagate
        s_img = (source_tensor + 1) / 2.0
        t_img = (adv_tensor + 1) / 2.0
        lpips_val = lpips_loss(s_img, t_img)

        perceptual_penalty = torch.relu(lpips_val - lpips_threshold)
        total_loss = latent_loss + perceptual_penalty

        # Gradient step: signed update
        grad = torch.autograd.grad(total_loss, modifier, retain_graph=False)[0]
        modifier = modifier - actual_step_size * torch.sign(grad)

        # Clamp and detach modifier to prevent graph accumulation
        modifier = torch.clamp(modifier, -max_change, max_change).detach()

        if (i % 50 == 0 or i == t_size - 1) and verbose:
            print(f"[{i}/{t_size}]\tLatent Loss: {latent_loss.item():.4f}\tLPIPS: {lpips_val.item():.4f}")

    return modifier


