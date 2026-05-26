import argparse
import csv
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from baseline_utils.wgan_helper import Variable, dtype
from baseline_utils.wgan_gp import build_generator, load_dft_basis
from loaders import _load_mat_file, expand_scenarios, resolve_dataset_path


def load_channels_for_scenarios(scenario_list, seed, num_paths, max_samples=None):
    channels = None
    filenames = []
    for scenario in scenario_list:
        filename = resolve_dataset_path(scenario, seed, num_paths=num_paths)
        filenames.append(filename)
        contents = _load_mat_file(filename)
        scenario_channels = np.asarray(contents['channels'], dtype=np.complex64)
        if max_samples is not None:
            scenario_channels = scenario_channels[:max_samples]
        scenario_channels = scenario_channels.transpose((1, 2, 0))
        if channels is None:
            channels = scenario_channels
        else:
            channels = np.concatenate((channels, scenario_channels), axis=-1)
    return channels, filenames


def phase_levels_for_dimension(size):
    return 2 ** int(math.ceil(math.log2(max(size, 1))))


def training_precoder(n_tx, num_streams, phase_levels):
    angles = np.linspace(0, 2 * np.pi, phase_levels, endpoint=False)
    angle_index = np.random.choice(len(angles), (n_tx, num_streams))
    return (1 / np.sqrt(n_tx)) * np.exp(1j * angles[angle_index])


def training_combiner(n_rx, num_rx_rf, phase_levels):
    angles = np.linspace(0, 2 * np.pi, phase_levels, endpoint=False)
    angle_index = np.random.choice(len(angles), (n_rx, num_rx_rf))
    weights = (1 / np.sqrt(n_rx)) * np.exp(1j * angles[angle_index])
    return np.matrix(weights).getH()


def format_snr_value(snr):
    if float(snr).is_integer():
        return int(snr)
    return float(snr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--train', type=str, default='Rural')
    parser.add_argument('--test', type=str, default='Rural')
    parser.add_argument('--num_paths', type=int, default=10)
    parser.add_argument('--spacing', nargs='+', type=float, default=[0.5])
    parser.add_argument('--pilot_alpha', type=float, default=0.6)
    parser.add_argument('--train_seed', type=int, default=1111)
    parser.add_argument('--val_seed', type=int, default=2222)
    parser.add_argument('--snr_values', nargs='+', type=float, default=[-15, -10, -5, 0, 5, 10, 15, 20])
    parser.add_argument('--model_iter', type=int, default=58000)
    parser.add_argument('--latent_dim', type=int, default=65)
    parser.add_argument('--nrepeat', type=int, default=100)
    parser.add_argument('--ntest', type=int, default=5)
    parser.add_argument('--latent_steps', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.02)
    parser.add_argument('--lambda_reg', type=float, default=1e-3)
    config = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(config.gpu)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True

    train_scenario_list = expand_scenarios(config.train)
    test_scenario_list = expand_scenarios(config.test)
    config.train_scenario_list = train_scenario_list
    config.test_scenario_list = test_scenario_list

    train_channels, train_files = load_channels_for_scenarios(
        train_scenario_list,
        config.train_seed,
        config.num_paths,
    )
    n_rx, n_tx = train_channels.shape[0], train_channels.shape[1]
    if n_tx % 4 != 0 or n_rx % 4 != 0:
        raise ValueError(f'WGAN generator requires N_t and N_r divisible by 4, got N_t={n_tx}, N_r={n_rx}.')

    mu_train = np.zeros([1])
    std_train = np.std(train_channels)

    test_channels, test_files = load_channels_for_scenarios(
        test_scenario_list,
        config.val_seed,
        config.num_paths,
        max_samples=config.nrepeat,
    )
    if test_channels.shape[0] != n_rx or test_channels.shape[1] != n_tx:
        raise ValueError(
            f'Train/test channel dimensions do not match: '
            f'train N_r={n_rx}, N_t={n_tx}; '
            f'test N_r={test_channels.shape[0]}, N_t={test_channels.shape[1]}.'
        )
    if test_channels.shape[2] < config.nrepeat:
        raise ValueError(
            f'Only {test_channels.shape[2]} test samples are available, '
            f'but nrepeat={config.nrepeat}.'
        )
    test_channels = (test_channels - mu_train) / std_train

    length = n_tx // 4
    breadth = n_rx // 4
    generator = build_generator(config.latent_dim, 1, length, breadth)
    checkpoint = os.path.join(PROJECT_DIR, f'models/wgan_gp/{config.train}/generator{config.model_iter}.pt')
    generator.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    generator.eval()

    a_t, a_r = load_dft_basis(n_tx, n_rx)
    a_t_r = np.kron(a_t.conj(), a_r)
    a_t_r_real = dtype(np.real(a_t_r))
    a_t_r_imag = dtype(np.imag(a_t_r))

    num_streams = min(n_tx, n_rx)
    num_rx_rf = n_rx
    num_pilots = max(1, int(math.floor(config.pilot_alpha * n_tx)))
    tx_phase_levels = phase_levels_for_dimension(n_tx)
    rx_phase_levels = phase_levels_for_dimension(n_rx)

    snr_vec = np.asarray(config.snr_values, dtype=float)
    nmse_all = np.zeros((len(snr_vec), config.nrepeat, config.ntest))
    qpsk_constellation = (1 / np.sqrt(2)) * np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j])

    pilot_sequence_ind = np.random.randint(0, 4, size=(num_streams, num_pilots))
    symbols = qpsk_constellation[pilot_sequence_ind]
    precoder_training = training_precoder(n_tx, num_streams, tx_phase_levels)
    combiner = training_combiner(n_rx, num_rx_rf, rx_phase_levels)
    sensing_matrix = np.kron(np.matmul(symbols.T, precoder_training.T), combiner)

    sensing_real = dtype(np.real(sensing_matrix))
    sensing_imag = dtype(np.imag(sensing_matrix))

    print(f'Loaded WGAN train data from: {train_files}')
    print(f'Loaded WGAN test data from: {test_files}')
    print(f'N_t={n_tx}, N_r={n_rx}, streams={num_streams}, rx_rf={num_rx_rf}, pilots={num_pilots}')
    print(f'Using checkpoint: {checkpoint}')

    for ind in range(config.nrepeat):
        vec_h_single = np.reshape(test_channels[:, :, ind].flatten('F'), [n_rx * n_tx, 1])
        noiseless_signal = np.matmul(test_channels[:, :, ind], np.matmul(precoder_training, symbols))
        signal_power = np.multiply(noiseless_signal, np.conj(noiseless_signal))

        for snr_idx, snr in enumerate(snr_vec):
            for test_idx in range(config.ntest):
                noise_matrix = (1 / np.sqrt(2)) * (
                    np.random.randn(n_rx, num_pilots) + 1j * np.random.randn(n_rx, num_pilots)
                )
                std_dev = (1 / (10 ** (snr / 20))) * np.sqrt(signal_power)
                rx_signal = noiseless_signal + np.multiply(std_dev, noise_matrix)
                rx_signal = np.matmul(combiner, rx_signal)

                vec_y = np.zeros((num_rx_rf * num_pilots, 1, 1), dtype='complex64')
                vec_y[:, 0, 0] = rx_signal.flatten('F')
                vec_y_real = dtype(np.real(vec_y[:, :, 0]))
                vec_y_imag = dtype(np.imag(vec_y[:, :, 0]))

                def gen_output(latent):
                    pred = generator(latent)
                    pred_real = torch.mm(a_t_r_real, pred[0, 0, :, :].view(n_tx * n_rx, -1)) - torch.mm(
                        a_t_r_imag,
                        pred[0, 1, :, :].view(n_tx * n_rx, -1),
                    )
                    pred_imag = torch.mm(a_t_r_real, pred[0, 1, :, :].view(n_tx * n_rx, -1)) + torch.mm(
                        a_t_r_imag,
                        pred[0, 0, :, :].view(n_tx * n_rx, -1),
                    )
                    diff_real = vec_y_real - torch.mm(sensing_real, pred_real) + torch.mm(sensing_imag, pred_imag)
                    diff_imag = vec_y_imag - torch.mm(sensing_real, pred_imag) - torch.mm(sensing_imag, pred_real)
                    return torch.norm(diff_real) ** 2 + torch.norm(diff_imag) ** 2 + config.lambda_reg * torch.norm(latent) ** 2

                latent = Variable(torch.randn(1, config.latent_dim)).type(dtype)
                latent.requires_grad = True
                optimizer = torch.optim.Adam([latent], lr=config.learning_rate)
                for _ in range(config.latent_steps):
                    optimizer.zero_grad()
                    loss = gen_output(latent)
                    loss.backward()
                    optimizer.step()

                gen_imgs = generator(latent).data.cpu().numpy()
                gen_imgs_complex = gen_imgs[0, 0, :, :] + 1j * gen_imgs[0, 1, :, :]
                gen_imgs_complex = np.matmul(a_t_r, np.reshape(gen_imgs_complex, [n_tx * n_rx, 1]))
                nmse = np.sum(np.square(np.abs(gen_imgs_complex - vec_h_single))) / np.sum(np.square(np.abs(vec_h_single)))
                nmse_all[snr_idx, ind, test_idx] = nmse
                print(snr, config.model_iter, nmse)

        print(f'sample {ind + 1}/{config.nrepeat}, current avg NMSE: {nmse_all.min(axis=-1)[:, :ind + 1].mean(axis=-1)}')

    wgan_nmse = nmse_all.min(axis=-1).mean(axis=-1)

    result_dir = os.path.join(PROJECT_DIR, f'results/wgan_gp/train{config.train}_test{config.test}')
    os.makedirs(result_dir, exist_ok=True)

    plt.rcParams['font.size'] = 14
    plt.figure(figsize=(10, 10))
    plt.plot(snr_vec, wgan_nmse, linewidth=4, label='test scenario: %s' % config.test)
    plt.grid()
    plt.legend()
    plt.title('WGAN based channel estimation')
    plt.xlabel('SNR [dB]')
    plt.ylabel('NMSE')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'results_mse.png'), dpi=300, bbox_inches='tight')
    plt.close()

    csv_rows = [['SNR', 'WGAN']]
    for snr, nmse in zip(snr_vec, wgan_nmse):
        csv_rows.append([format_snr_value(snr), float(nmse)])

    with open(os.path.join(result_dir, 'results.csv'), 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['SNR', 'wgan'])
        for snr, nmse in zip(snr_vec, wgan_nmse):
            writer.writerow([format_snr_value(snr), float(nmse)])

    save_dict = {
        'nmse_all': nmse_all,
        'avg_nmse': wgan_nmse,
        'pilot_alpha': config.pilot_alpha,
        'snr_range': snr_vec,
        'config': config,
        'checkpoint': checkpoint,
        'n_tx': n_tx,
        'n_rx': n_rx,
    }
    torch.save(save_dict, os.path.join(result_dir, 'results.pt'))

    print(csv_rows)


if __name__ == '__main__':
    main()
