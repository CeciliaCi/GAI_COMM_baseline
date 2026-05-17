#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import hdf5storage  # type: ignore
except ImportError:  # pragma: no cover - depends on the runtime env
    hdf5storage = None

try:
    import scipy.io as sio  # type: ignore
except ImportError:  # pragma: no cover - depends on the runtime env
    sio = None
sys.path.append('../')
PARENT_DIR = os.path.dirname(os.path.realpath(__file__))
#ROOT_DIR = os.path.dirname(PROJECT_DIR)
ROOT_DIR = os.path.dirname(PARENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

DEFAULT_DEEPMIMO_SCENARIOS = ['O1_28B', 'O1_28', 'I2_28B']
DEFAULT_LEO_SCENARIOS = ['Rural', 'Urban', 'DenseUrban']


def _load_mat_file(filename):
    if hdf5storage is not None:
        return hdf5storage.loadmat(filename)
    if sio is not None:
        return sio.loadmat(filename)
    raise ImportError(
        "Neither `hdf5storage` nor `scipy.io` is available. "
        "Install one of them to read `.mat` channel datasets."
    )


def normalize_scenario_name(scenario):
    """Map accepted aliases to the canonical scenario name used by this repo."""
    if scenario.startswith('LEO_'):
        scenario = scenario[len('LEO_'):]
    return scenario


def expand_scenarios(train_or_test):
    """Expand aggregate scenario names into explicit scenario lists."""
    if train_or_test == 'mixed':
        return list(DEFAULT_DEEPMIMO_SCENARIOS)
    if train_or_test in ('mixed_leo', 'LEO_mixed'):
        return list(DEFAULT_LEO_SCENARIOS)
    return [normalize_scenario_name(train_or_test)]


def resolve_dataset_path(scenario, seed, num_paths=10):
    """Return the dataset path for either DeepMIMO or LEO naming conventions."""
    scenario = normalize_scenario_name(scenario)
    leo_filename = os.path.join(PARENT_DIR, f'dataset/LEO_{scenario}_seed{seed}.mat')
    deepmimo_filename = os.path.join(
        PARENT_DIR,
        f'DeepMIMO-5GNR/DeepMIMO_dataset/{scenario}_path{num_paths}_seed{seed}.mat',
    )
    candidates = [leo_filename, deepmimo_filename]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find dataset for scenario `{scenario}` and seed `{seed}`. "
        f"Tried: {candidates}"
    )


def load_channel_array(scenario_list, seed, num_paths):
    channels = np.array([], dtype='complex64')
    filenames = []
    for scenario in scenario_list:
        filename = resolve_dataset_path(scenario, seed, num_paths=num_paths)
        filenames.append(filename)
        contents = _load_mat_file(filename)
        channel_scenario = np.asarray(contents['channels'], dtype=np.complex64)
        if len(channels) < 1:
            channels = channel_scenario
        else:
            channels = np.concatenate((channels, channel_scenario), axis=0)
    channels = np.asarray(channels)
    channels = np.reshape(channels, (-1, channels.shape[-2], channels.shape[-1]))
    return channels, filenames


def infer_channel_image_size(scenario_list, seed=1111, num_paths=10):
    channels, _ = load_channel_array(scenario_list, seed, num_paths)
    n_rx, n_tx = channels.shape[-2], channels.shape[-1]
    return [n_tx, n_rx]

def transpose_3d(x):
    return x.transpose([0, 2, 1])


class Channels(Dataset):
    """Load DeepMIMO Channels"""

    # def __init__(self, seed, config, norm=None, gan=False):
    def __init__(self, seed, config, norm=None):
        """
        @:param
            seed: Seed used to generate the target DeepMIMO dataset
            config: Configuration of MIMO channels
            norm: Normalization method
        @:return A torch dataset for sampling
        """

        ################# Parameters of channels #############
        self.num_paths = config.data.num_paths
        self.spacings = np.copy(config.data.spacing_list)
        self.filenames = []
        # self.gan = gan
        ################ Load channels from files ############
        self.channels, self.filenames = load_channel_array(
            config.data.scenario_list,
            seed,
            self.num_paths,
        )
        n_rx, n_tx = self.channels.shape[-2], self.channels.shape[-1]
        self.n_tx, self.n_rx = n_tx, n_rx
        config.data.image_size = [n_tx, n_rx]
        # Channel normalization
        if norm == 'global':
            self.mean = 0.
            self.std = np.std(self.channels)
        elif norm == 'zero_mean':
            self.mean = np.mean(self.channels)
            self.std = np.std(self.channels)
        elif type(norm) == list:
            self.mean = norm[0]
            self.std = norm[1]
        else:
            self.mean = 0.
            self.std = np.std(self.channels)

        # Sample random QPSK pilots
        self.pilots = 1 / np.sqrt(2) * (2 * np.random.binomial(1, 0.5, size=(
            self.channels.shape[0], n_tx, config.data.num_pilots)) - 1 + 1j * (2 * np.random.binomial(1, 0.5, size=(
            self.channels.shape[0], n_tx, config.data.num_pilots)) - 1))

        # SNR and noise power
        self.SNR_range = np.array([-10, 30])
        self.noise_power_range = 10 ** (-self.SNR_range / 10) * n_tx

        # if gan:
        #     self.H_train = self.get_gan_data(config)

    # def get_gan_data(self, config):
    #     N_t, N_r = self.n_tx, self.n_rx
    #     dft_basis = sio.loadmat("data\dft_basis.mat")
    #     A_T = dft_basis['A1'] / np.sqrt(N_t)
    #     A_R = dft_basis['A2'] / np.sqrt(N_r)
    #     H = self.channels  # [n, N_r, N_t]
    #     H_extracted = transpose_3d(H)  # [n, N_t, N_r]
    #     M_left = np.array([A_R.conj().T for _ in range(H.shape[0])])
    #     M_right = np.array([A_T for _ in range(H.shape[0])])
    #     H_extracted = transpose_3d(np.matmul(np.matmul(M_left, transpose_3d(H_extracted)), M_right))  # [n, N_t, N_r]
    #     img_np_real = H_extracted.real  # [n, N_t, N_r]
    #     img_np_imag = H_extracted.imag  # [n, N_t, N_r]
    #     mu_real = np.mean(img_np_real, axis=0)  # [N_t, N_r]
    #     mu_imag = np.mean(img_np_imag, axis=0)
    #     std_real = np.std(img_np_real, axis=0)
    #     std_imag = np.std(img_np_imag, axis=0)
    #     self.H_ext_mu = [mu_real, mu_imag]
    #     self.H_ext_std = [std_real, std_imag]
    #     img_np_real = (img_np_real - mu_real) / std_real.reshape([1, N_t, N_r])
    #     img_np_imag = (img_np_imag - mu_imag) / std_imag.reshape([1, N_t, N_r])  # [n, N_t, N_r]
    #     return np.asarray(np.stack((img_np_real, img_np_imag), 1))  # [n, 2, N_t, N_r]

    def __len__(self):
        return len(self.channels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Fetch one batch of channel samples
        H = self.channels[idx]

        # Normalize
        H_normalized = (H - self.mean) / self.std
        # Obtain both real and image components
        H_normalized_coefficient = np.stack((np.real(H_normalized), np.imag(H_normalized)), axis=0)
        
        # Hermitian of normalized channels
        H_normalized_herm = np.conj(np.transpose(H_normalized))
        H_normalized_herm_coefficient = np.stack((np.real(H_normalized_herm), np.imag(H_normalized_herm)), axis=0)

        # Obtain pilots (complex)
        P = self.pilots[idx]
        P_herm = np.conj(np.transpose(P))

        # Sample noisy signal observation Y
        Y = np.matmul(H_normalized, P)
        noise_power = self.noise_power_range[1] + (
                    self.noise_power_range[0] - self.noise_power_range[1]) * np.random.rand(1)
        N = np.sqrt(noise_power) * (np.random.normal(size=Y.shape) + 1j * np.random.normal(size=Y.shape))
        Y = Y + N
        Y_herm = np.conj(np.transpose(Y))

        sample = {'H': H_normalized_coefficient.astype(np.float32),  # (batch_size, 2, n_rx, n_tx)
                  'H_herm': H_normalized_herm_coefficient.astype(np.float32),  # (batch_size, 2, n_tx, n_rx)
                  'P': self.pilots[idx].astype(np.complex64),
                  'P_herm': P_herm.astype(np.complex64),
                  'Y': Y.astype(np.complex64),
                  'Y_herm': Y_herm.astype(np.complex64),
                  'sigma_n': noise_power.astype(np.float32),
                  'idx': int(idx)}

        # if self.gan:
        #     sample = {'H': H_normalized_coefficient.astype(np.float32),
        #               'H_train': self.H_train[idx].astype(np.float32),  # [n_batch, 2, n_rx, n_tx]
        #               'P': self.pilots[idx].astype(np.complex64),
        #               'Y': Y.astype(np.complex64),
        #               'sigma_n': self.noise_power.astype(np.float32),
        #               'idx': int(idx)}

        return sample


def map_complex_to_components(X):
    """
    :param X: a tensor of complex channels (batch_size, n_tx, n_rx)
    :return: {X.real, X.img} shape:(batch_size, 2, n_tx, n_rx)
    """

    X = torch.view_as_real(X)  # (batch_size, n_tx, n_rx, 2)
    return X.permute([0, 3, 1, 2])


def map_components_to_complex(X):
    """
    :param X={X.real, X.img}
    :param X.shape is (batch_size, 2, n_rx, n_tx)
    :return: Complex tensors in shape (batch_size, w, h)
    """
    X = X.permute([0, 2, 3, 1])
    return X[:, :, :, 0] + 1j * X[:, :, :, 1]
