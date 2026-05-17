import argparse
import os
import sys

import numpy as np
import torch
import torch.autograd as autograd
import torch.optim as optim

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

try:
    from baseline_utils.wgan_helper import Conv2d, Variable, View, dtype
except ImportError:
    from wgan_helper import Conv2d, Variable, View, dtype

from loaders import _load_mat_file, expand_scenarios, load_channel_array

reset_optim_D = True

# Disable TF32 due to potential precision issues
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = True


def make_dft_matrix(size):
    row = np.arange(size).reshape(-1, 1)
    col = np.arange(size).reshape(1, -1)
    return np.exp(-2j * np.pi * row * col / size).astype(np.complex64) / np.sqrt(size)


def load_dft_basis(n_tx, n_rx):
    dft_basis = _load_mat_file(os.path.join(PROJECT_DIR, 'data/dft_basis.mat'))
    a_t = dft_basis.get('A1')
    a_r = dft_basis.get('A2')
    if a_t is None or a_t.shape != (n_tx, n_tx):
        a_t = make_dft_matrix(n_tx)
    else:
        a_t = a_t / np.sqrt(n_tx)
    if a_r is None or a_r.shape != (n_rx, n_rx):
        a_r = make_dft_matrix(n_rx)
    else:
        a_r = a_r / np.sqrt(n_rx)
    return a_t, a_r


def load_training_tensor(scenario_list, seed, num_paths, n_tx=None, n_rx=None):
    channels, filenames = load_channel_array(scenario_list, seed, num_paths)
    channels = np.transpose(channels, (1, 2, 0))  # [N_r, N_t, n]
    actual_n_rx, actual_n_tx = channels.shape[0], channels.shape[1]
    n_tx = actual_n_tx if n_tx is None else n_tx
    n_rx = actual_n_rx if n_rx is None else n_rx
    if (n_rx, n_tx) != (actual_n_rx, actual_n_tx):
        raise ValueError(
            f'Configured WGAN dimensions (N_r={n_rx}, N_t={n_tx}) do not match '
            f'loaded channels (N_r={actual_n_rx}, N_t={actual_n_tx}).'
        )
    mu = np.zeros([1], dtype=np.float32)
    std = np.std(channels)
    channels = (channels - mu) / std

    h_extracted = np.transpose(channels.copy(), (2, 1, 0))  # [n, N_t, N_r]
    a_t, a_r = load_dft_basis(n_tx, n_rx)
    for idx in range(h_extracted.shape[0]):
        h_extracted[idx] = np.transpose(
            np.matmul(np.matmul(a_r.conj().T, h_extracted[idx].T), a_t).astype(np.complex64)
        )

    img_np = np.zeros((channels.shape[2], 2, n_tx, n_rx), dtype=np.float32)
    img_np[:, 0, :, :] = np.real(h_extracted)
    img_np[:, 1, :, :] = np.imag(h_extracted)
    return img_np, filenames, float(std)


def build_generator(latent_dim, batch_size, length, breadth):
    generator = torch.nn.Sequential(
        torch.nn.Linear(latent_dim, 128 * length * breadth),
        torch.nn.ReLU(),
        View([batch_size, 128, length, breadth]),
        torch.nn.Upsample(scale_factor=2),
        Conv2d(128, 128, 4, bias=False),
        torch.nn.BatchNorm2d(128, momentum=0.8),
        torch.nn.ReLU(),
        torch.nn.Upsample(scale_factor=2),
        Conv2d(128, 128, 4, bias=False),
        torch.nn.BatchNorm2d(128, momentum=0.8),
        torch.nn.ReLU(),
        Conv2d(128, 2, 4, bias=False),
    )
    return generator.type(dtype)


def build_discriminator(n_tx=64, n_rx=16):
    feature_layers = torch.nn.Sequential(
        Conv2d(2, 16, 3, stride=2),
        torch.nn.LeakyReLU(0.2, inplace=True),
        torch.nn.Dropout(0.25),
        Conv2d(16, 32, 3, stride=2),
        torch.nn.ZeroPad2d(padding=(0, 1, 0, 1)),
        torch.nn.LeakyReLU(0.2, inplace=True),
        torch.nn.Dropout(0.25),
        Conv2d(32, 64, 3, stride=2),
        torch.nn.LeakyReLU(0.2, inplace=True),
        torch.nn.Dropout(0.25),
        Conv2d(64, 128, 3, stride=1),
        torch.nn.LeakyReLU(0.2, inplace=True),
        torch.nn.Dropout(0.25),
    )
    with torch.no_grad():
        dummy = torch.zeros(1, 2, n_tx, n_rx)
        flatten_dim = feature_layers(dummy).reshape(1, -1).shape[1]
    discriminator = torch.nn.Sequential(
        feature_layers,
        torch.nn.Flatten(),
        torch.nn.Linear(flatten_dim, 1),
    )
    return discriminator.type(dtype)


def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN-GP."""
    alpha = dtype(np.random.random((real_samples.size(0), 1, 1, 1)))
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    fake = Variable(dtype(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = torch.reshape(gradients, (gradients.size(0), -1))
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--train', type=str, default='mixed')
    parser.add_argument('--num_paths', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--n_iters', type=int, default=60000)
    parser.add_argument('--critic_steps', type=int, default=5)
    parser.add_argument('--latent_dim', type=int, default=65)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--lambda_gp', type=float, default=10.0)
    parser.add_argument('--save_freq', type=int, default=2000)
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    scenario_list = expand_scenarios(args.train)
    model_dir = os.path.join(PROJECT_DIR, 'models', 'wgan_gp', args.train)
    os.makedirs(model_dir, exist_ok=True)

    train_seed = 1111
    x_train, filenames, train_std = load_training_tensor(
        scenario_list=scenario_list,
        seed=train_seed,
        num_paths=args.num_paths,
    )
    n_tx, n_rx = x_train.shape[-2], x_train.shape[-1]
    length = n_tx // 4
    breadth = n_rx // 4
    print(f'Loaded {x_train.shape[0]} training samples from: {filenames}')
    print(f'Normalized WGAN training tensor shape: {x_train.shape}; std={train_std:.6f}')

    batch_size = min(args.batch_size, x_train.shape[0])
    replace = x_train.shape[0] < args.batch_size
    generator = build_generator(args.latent_dim, batch_size, length, breadth)
    discriminator = build_discriminator(n_tx, n_rx)

    def reset_grad():
        generator.zero_grad()
        discriminator.zero_grad()

    g_solver = optim.RMSprop(generator.parameters(), lr=args.lr)
    d_solver = optim.RMSprop(discriminator.parameters(), lr=args.lr)

    for iteration in range(args.n_iters):
        if reset_optim_D:
            d_solver = optim.RMSprop(discriminator.parameters(), lr=args.lr)

        for _ in range(args.critic_steps):
            z = Variable(torch.randn(batch_size, args.latent_dim)).type(dtype)
            idx = np.random.choice(x_train.shape[0], batch_size, replace=replace)
            x = Variable(torch.from_numpy(x_train[idx]).float()).type(dtype)

            g_sample = generator(z)
            d_real = discriminator(x)
            d_fake = discriminator(g_sample)
            gradient_penalty = compute_gradient_penalty(discriminator, x, g_sample)
            d_loss = -(torch.mean(d_real) - torch.mean(d_fake)) + args.lambda_gp * gradient_penalty

            d_loss.backward()
            d_solver.step()
            reset_grad()

        z = Variable(torch.randn(batch_size, args.latent_dim)).type(dtype)
        g_sample = generator(z)
        d_fake = discriminator(g_sample)
        g_loss = -torch.mean(d_fake)

        g_loss.backward()
        g_solver.step()
        reset_grad()

        if iteration % args.save_freq == 0:
            torch.save(generator.state_dict(), os.path.join(model_dir, f'generator{iteration}.pt'))

        if iteration % 50 == 0:
            print(
                f'Iter-{iteration}; '
                f'D_loss: {d_loss.detach().cpu().item():.6f}; '
                f'G_loss: {g_loss.detach().cpu().item():.6f}'
            )

    torch.save(generator.state_dict(), os.path.join(model_dir, 'final_model.pt'))
