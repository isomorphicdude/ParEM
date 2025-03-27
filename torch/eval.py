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
import torchvision
import numpy as np

# Import custom modules
from parem.algorithms import Algorithm, PGD, ShortRun, VI, AlternatingBackprop
from parem.models import NLVM  # Generator network model
from parem.utils import get_mnist, load_model_ckpt
from parem.stats import compute_fid

MODEL_CKPT_PATH = "/content/ParEM/checkpoints"
MODEL_IMG_PATH = "/content/ParEM/images"
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

# PGD Settings
STEP_SIZE = 1e-4  # h: step size
LAMBDA = 1e-3 / (STEP_SIZE * N_IMAGES)  # lambda
N_PARTICLES = 10  # N: number of particles

# VAE settings
VAE_Q_OPTIM = "adam"
VAE_THETA_OPTIM = "rmsprop"
VAE_Q_LR = 1e-3
VAE_THETA_LR = 1e-3
CHAIN_PARAM = False # whether to use the same optimizer for q and theta


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
@click.option(
    "--sigma2",
    type=float,
    default=1e-4,
    help="Likelihood variance",
)
@click.option(
    "--kl_coeff",
    type=float,
    default=1.0,
    help="KL coefficient",
)
@click.option(
    "--use-new-arch",
    is_flag=True,
    default=False,
    help="Use new architecture for the generator",
)
@click.option(
    "--just-train",
    is_flag=True,
    default=False,
    help="Just train the model without evaluation",
)
def run(name, task, sigma2, kl_coeff, use_new_arch, just_train):
    click.echo(f"Running {name} on {task} task")
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Load MNIST dataset
    mnist_train = get_mnist(DATASET_PATH, N_IMAGES, train=True)
    mnist_test = get_mnist(DATASET_PATH, N_IMAGES, train=False)

    # Initialize the NLVM model
    generator = NLVM(x_dim=X_DIM, sigma2=sigma2, nc=1).to(DEVICE)

    lvm = get_model(name, generator, mnist_train, kl_coeff=kl_coeff, use_new_arch=use_new_arch)
    
    os.makedirs(MODEL_CKPT_PATH, exist_ok=True)
    os.makedirs(MODEL_IMG_PATH, exist_ok=True)
    
    model_save_path = os.path.join(
        MODEL_CKPT_PATH,
        f"{name}_algo.pth"
    )

    metrics_file = os.path.join(
        MODEL_CKPT_PATH,
        f"{name}_metrics.yaml"
    )

    # load checkpoint
    try:
        print(f"Loading checkpoint from {model_save_path}")
        load_model_ckpt(model_save_path, lvm)
        if just_train:
            raise FileNotFoundError
    except FileNotFoundError:
        print(f"Checkpoint not found at {model_save_path}")
        lvm.run(N_EPOCHS,
                model_save_path,
                wandb_log=False,
                log_images=False,
                compute_stats=False)

    
    metrics = {'model': name}
    
    if just_train:
        task = None
        
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
        print(f"Train MSE: {mse_train}, Val MSE: {mse_val}")
    else:
        print("No task specified. Skipping evaluation.")
        return

    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            try:
                previous_results = yaml.safe_load(f)
                if not isinstance(previous_results, list):
                    previous_results = [previous_results]
            except Exception:
                previous_results = []
    else:
        previous_results = []
    previous_results.append(metrics)
    with open(metrics_file, "w") as f:
        yaml.dump(previous_results, f)
        
    print(f"Metrics saved to {metrics_file}")



def get_fid(lvm, n_samples, save_samples=False):
    """
    Compute fid between model and data samples (randomly subsampled)
    """
    name = lvm.__class__.__name__.lower()
    print(f"Computing FID for {name}")
    
    idx = torch.randint(0, len(lvm.dataset), size=(n_samples,))
    print("Synthesizing images...")
    gmm_samples = lvm.synthesize_images(n_samples,
                                           show=False,
                                           approx_type='gmm')
    data_samples = torch.stack([lvm.dataset[_id][0]
                                for _id in idx], dim=0)
    
    print("Computing FID...")
    gmm_fid = compute_fid(data_samples,gmm_samples,nn_feature=None)

    # Images sampled from prior
    stdg_samples, _ = lvm._model.sample(n_samples)
    stdg_fid = compute_fid(data_samples,stdg_samples,nn_feature=None)

    if save_samples:
        torch.save(data_samples, os.path.join(MODEL_IMG_PATH, f"{name}_data_samples.pt"))
        torch.save(gmm_samples, os.path.join(MODEL_IMG_PATH, f"{name}_gmm_samples.pt"))
        torch.save(stdg_samples, os.path.join(MODEL_IMG_PATH, f"{name}_stdg_samples.pt"))
        
    # plot and save images
    gmm_grid = torchvision.utils.make_grid(gmm_samples[:100,...], nrow=10)
    stdg_grid = torchvision.utils.make_grid(stdg_samples[:100, ...], nrow=10)
    torchvision.utils.save_image(gmm_grid, os.path.join(MODEL_IMG_PATH, f"{name}_gmm_samples.png"))
    torchvision.utils.save_image(stdg_grid, os.path.join(MODEL_IMG_PATH, f"{name}_stdg_samples.png"))
    
    return gmm_fid, stdg_fid


def get_inpaint(lvm, n_samples, val_dataset, batch_size=100, mask=None):
    name = lvm.__class__.__name__.lower()
    print(f"Computing inpainting for {name}")
    indices = torch.randint(0, len(val_dataset), size=(n_samples,))
    mse_total = 0.0
    num_batches = (n_samples + batch_size - 1) // batch_size  # ceiling division
    for b in range(num_batches):
        print(f"Batch {b+1}/{num_batches}")
        batch_indices = indices[b*batch_size:(b+1)*batch_size]
        batch_imgs = torch.stack([val_dataset[idx][0] for idx in batch_indices])
        batch_reconstructed = lvm.reconstruct(batch_imgs, mask, show=False)
        batch_mse = ((batch_imgs - batch_reconstructed) ** 2).mean(dim=[1,2,3])
        mse_total += batch_mse.sum().item()

    mse = mse_total / n_samples
    
    # plot and save images
    true_grid = torchvision.utils.make_grid(batch_imgs, nrow=10)
    recon_grid = torchvision.utils.make_grid(batch_reconstructed, nrow=10)
    torchvision.utils.save_image(true_grid, os.path.join(MODEL_IMG_PATH, f"{name}_true.png"))
    torchvision.utils.save_image(recon_grid, os.path.join(MODEL_IMG_PATH, f"{name}_recon.png"))
    
    return mse


def get_recon(lvm, n_samples, train_dataset, val_dataset, batch_size=100):
    #TODO: add FID
    # for no masks on val and train
    name = lvm.__class__.__name__.lower()
    print(f"Computing reconstruction for {name}")
    train_indices = torch.randint(0, len(train_dataset), size=(n_samples,))
    val_indices = torch.randint(0, len(val_dataset), size=(n_samples,))
    num_batches = (n_samples + batch_size - 1) // batch_size  # ceiling division
    
    mse_train = 0.0
    mse_val = 0.0
    
    for b in range(num_batches):
        print(f"Batch {b+1}/{num_batches}")
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
    
    # plot and save images
    train_true_grid = torchvision.utils.make_grid(train_batch_imgs, nrow=10)
    train_recon_grid = torchvision.utils.make_grid(train_batch_reconstructed, nrow=10)
    val_true_grid = torchvision.utils.make_grid(val_batch_imgs, nrow=10)
    val_recon_grid = torchvision.utils.make_grid(val_batch_reconstructed, nrow=10)
    
    torchvision.utils.save_image(train_true_grid, os.path.join(MODEL_IMG_PATH, f"{name}_train_true.png"))
    torchvision.utils.save_image(train_recon_grid, os.path.join(MODEL_IMG_PATH, f"{name}_train_recon.png"))
    torchvision.utils.save_image(val_true_grid, os.path.join(MODEL_IMG_PATH, f"{name}_val_true.png"))
    torchvision.utils.save_image(val_recon_grid, os.path.join(MODEL_IMG_PATH, f"{name}_val_recon.png"))
    
    return mse_train, mse_val

def get_model(name, generator, dataset, chain_param=CHAIN_PARAM, kl_coeff=1.0, use_new_arch=False) -> Algorithm:
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
            use_common_optimizer=chain_param,
            kl_coeff=kl_coeff,
            use_new_arch=use_new_arch,
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
