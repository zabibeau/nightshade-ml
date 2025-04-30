import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def fgsm_penalty(source_tensor, target_latent, modifier, latent_function, eps=0.25):
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

def pgd_penalty(source_tensor, target_latent, modifier, latent_function, iterations=50, step_size=0.03, eps=0.2):
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
    t_size=150,
    eps=0.15,
):
    # Constants
    max_change = eps / 0.5
    step_size = max_change

    # Make sure all tensors are detached and clean
    source_tensor = source_tensor.detach()
    target_latent = target_latent.detach()
    modifier = modifier.detach()

    for i in range(t_size):
        # Adjust step size (optional decay)
        actual_step_size = step_size - (step_size - step_size / 100) * (i / t_size)

        # Reattach modifier with gradients enabled
        modifier.requires_grad_(True)

        # Create adversarial input
        adv_tensor = torch.clamp(modifier + source_tensor, -1.0, 1.0)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            adv_latent = latent_function(adv_tensor)

        # Compute MSE loss in latent space
        loss = torch.nn.functional.mse_loss(adv_latent, target_latent)

        # Backpropagate
        grad = torch.autograd.grad(loss, modifier, retain_graph=False, create_graph=False)[0]

        # Gradient step: signed update
        modifier = modifier - actual_step_size * torch.sign(grad)

        # Clamp and detach modifier to prevent graph accumulation
        modifier = torch.clamp(modifier, -max_change, max_change).detach()

        if i % 50 == 0 or i == t_size - 1:
            print(f"[{i}/{t_size}] Loss: {loss.item():.4f}")

    return modifier


