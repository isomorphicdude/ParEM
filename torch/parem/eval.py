"""
Evaluate the algorithms on reconstruction and generation.
Only MNIST for now.
"""

import os
import sys
import yaml
import random

import click
import torch
import numpy as np

# Import custom modules
from parem.algorithms import Algorithm
from parem.models import NLVM  # Generator network model
from parem.algorithms import PGD, ShortRun, VI, AlternatingBackprop
from parem.utils import get_mnist
from parem.stats import compute_fid


DATASET_PATH = "/content/ParEM/datasets/MNIST"
N_IMAGES = 10000
# Training settings
N_BATCH = 128  # M_b: batch size for theta updates
N_EPOCHS = 100  # n_epochs = K * M_b / M where K = total number of iterations
SEED = 1  # Seed for PRNG
# Device on which to carry out computations:
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OPTIMIZER = "rmsprop"  # Theta optimizer

# Model Settings
X_DIM = 64  # d_x: dimension of latent space
LIKELIHOOD_VAR = 0.01**2  # sigma^2

# PGD Settings
STEP_SIZE = 1e-4  # h: step size
LAMBDA = 1e-3 / (STEP_SIZE * N_IMAGES)  # lambda
N_PARTICLES = 10  # N: number of particles

# VAE settings
VAE_Q_OPTIM = "adam"
VAE_THETA_OPTIM = "rmsprop"
VAE_Q_LR = 1e-3
VAE_THETA_LR = 1e-3


@click.command()
@click.option(
    "--name",
    type=click.Choice(["pgd", "shortrun", "abp", "vae"]),
    default="pgd",
    help="Name of the model",
)
@click.option(
    "--task",
    type=click.Choice(["recon", "mask", "fid"]),
    default="recon",
    help="Task to run",
)
def run(name, task):
    click.echo(f"Running {name} on {task} task")
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Load MNIST dataset
    mnist_train = get_mnist(DATASET_PATH, N_IMAGES, train=True)
    mnist_test = get_mnist(DATASET_PATH, N_IMAGES, train=False)

    # Initialize the NLVM model
    generator = NLVM(x_dim=X_DIM, sigma2=LIKELIHOOD_VAR, nc=1).to(DEVICE)

    lvm = get_model(name, generator, mnist_train)

    model_save_path = os.path.join(
        "/content/ParEM/checkpoints",
        f"{name}_generator.pth"
    )

    metrics_file = os.path.join(
        "/content/ParEM/checkpoints",
        f"{name}_metrics.yaml"
    )

    # load checkpoint
    try:
        checkpoint = torch.load(model_save_path, weights_only=True)
        lvm._model.load_state_dict(checkpoint['generator'])
        lvm._posterior = checkpoint['particles']
    except FileNotFoundError:
        print(f"Checkpoint not found at {model_save_path}")
        lvm.run(N_EPOCHS,
                model_save_path,
                wandb_log=False,
                log_images=False,
                compute_stats=False)

    
    metrics = {'model': name}
        
    if task == "fid":
        gmm_fid, stdg_fid = get_fid(lvm, n_samples=1000, save_samples=False)
        metrics["gmm_fid"] = gmm_fid
        metrics["stdg_fid"] = stdg_fid
        print(f"GMM FID: {gmm_fid}, STDG FID: {stdg_fid}")
    elif task == "mask":
        mask = torch.ones(32, 32, dtype=torch.bool)
        for i in range(10, 22):
            for j in range(10, 22):
                mask[i, j] = False
        recon_mse = get_inpaint(lvm, n_samples=1000, val_dataset=mnist_test, batch_size=100, mask=mask)
        metrics["recon_mse"] = recon_mse
        print(f"Reconstruction MSE: {recon_mse}")
    elif task == "recon":
        mse_train, mse_val = get_recon(lvm, n_samples=1000, train_dataset=mnist_train, val_dataset=mnist_test, batch_size=100)
        metrics["mse_train"] = mse_train
        metrics["mse_val"] = mse_val
    else:
        raise ValueError(f"Unknown task: {task}")

    with open(metrics_file, "w") as f:
        yaml.dump(metrics, f)
        
    print(f"Metrics saved to {metrics_file}")

def get_fid(lvm, n_samples, save_samples=False):
    """
    Compute fid between model and data samples (randomly subsampled)
    """
    name = lvm._model.__class__.__name__.lower()
    idx = torch.randint(0, len(lvm.dataset), size=(n_samples,))
    gmm_samples = lvm.synthesize_images(n_samples,
                                           show=False,
                                           approx_type='gmm')
    data_samples = torch.stack([lvm.dataset[_id][0]
                                for _id in idx], dim=0)
    gmm_fid = compute_fid(data_samples,gmm_samples,nn_feature=None)

    # Images sampled from prior
    stdg_samples, _ = lvm._model.sample(n_samples)
    stdg_fid = compute_fid(data_samples,stdg_samples,nn_feature=None)

    if save_samples:
        torch.save(data_samples, f"{name}_data_samples.pt")
        torch.save(gmm_samples, f"{name}_gmm_samples.pt")
        torch.save(stdg_samples, f"{name}_stdg_samples.pt")
    return gmm_fid, stdg_fid


def get_inpaint(lvm, n_samples, val_dataset, batch_size=100, mask=None):
    indices = torch.randint(0, len(val_dataset), size=(n_samples,))
    mse_total = 0.0
    num_batches = (n_samples + batch_size - 1) // batch_size  # ceiling division
    for b in range(num_batches):
        batch_indices = indices[b*batch_size:(b+1)*batch_size]
        batch_imgs = torch.stack([val_dataset[idx][0] for idx in batch_indices])
        batch_reconstructed = lvm.reconstruct(batch_imgs, mask, show=False)
        batch_mse = ((batch_imgs - batch_reconstructed) ** 2).mean(dim=[1,2,3])
        mse_total += batch_mse.sum().item()

    mse = mse_total / n_samples
    return mse


def get_recon(lvm, n_samples, train_dataset, val_dataset, batch_size=100):
    #TODO: add FID
    # for no masks on val and train
    train_indices = torch.randint(0, len(train_dataset), size=(n_samples,))
    val_indices = torch.randint(0, len(val_dataset), size=(n_samples,))
    num_batches = (n_samples + batch_size - 1) // batch_size  # ceiling division
    
    mse_train = 0.0
    mse_val = 0.0
    
    for b in range(num_batches):
        train_batch_indices = train_indices[b*batch_size:(b+1)*batch_size]
        val_batch_indices = val_indices[b*batch_size:(b+1)*batch_size]
        
        # 
        train_batch_imgs = torch.stack([train_dataset[idx][0] for idx in train_batch_indices])
        val_batch_imgs = torch.stack([val_dataset[idx][0] for idx in val_batch_indices])
        
        train_batch_reconstructed = lvm.reconstruct(train_batch_imgs, show=False)
        val_batch_reconstructed = lvm.reconstruct(val_batch_imgs, show=False)
        
        train_batch_mse = ((train_batch_imgs - train_batch_reconstructed) ** 2).mean(dim=[1,2,3])
        val_batch_mse = ((val_batch_imgs - val_batch_reconstructed) ** 2).mean(dim=[1,2,3])
        
        mse_train += train_batch_mse.sum().item()
        mse_val += val_batch_mse.sum().item()
        
    mse_train /= n_samples
    mse_val /= n_samples
    
    return mse_train, mse_val

def get_model(name, generator, dataset) -> Algorithm:
    if name == "pgd":
        return get_pgd(PGD, generator, dataset)
    elif name == "shortrun":
        return get_pgd(ShortRun, generator, dataset)
    elif name == "abp":
        return get_pgd(AlternatingBackprop, generator, dataset)
    elif name == "vae":
        return VI(
            model=generator,
            dataset=dataset,
            train_batch_size=128,
            n_particles=N_PARTICLES,
            device=DEVICE,
            theta_optimizer=OPTIMIZER,
            q_optimizer=VAE_Q_OPTIM,
            theta_step_size=VAE_THETA_LR,
            q_step_size=VAE_Q_LR,
        )
    else:
        raise ValueError(f"Invalid model name: {name}")


def get_pgd(class_inst, generator, dataset) -> Algorithm:
    return class_inst(
        model=generator,
        dataset=dataset,
        train_batch_size=128,
        lambd=LAMBDA,
        n_particles=N_PARTICLES,
        particle_step_size=STEP_SIZE,
        device=DEVICE,
        theta_optimizer=OPTIMIZER,
    )
    
if __name__ == "__main__":
    run()
