# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import torch
import torchaudio

import random
from scipy.interpolate import interp1d
from numpy.fft import fft, ifft

TTransformIn = TypeVar("TTransformIn")
TTransformOut = TypeVar("TTransformOut")
Transform = Callable[[TTransformIn], TTransformOut]

def safe_tensor(t: torch.Tensor, nan: float = 0.0, posinf: float = 1e6, neginf: float = -1e6) -> torch.Tensor:
    """
    Replace NaN and Inf values in a tensor with specified numbers.
    Apply this only once at the end of a transform chain.
    """
    return torch.nan_to_num(t, nan=nan, posinf=posinf, neginf=neginf)

@dataclass
class ToTensor:
    """Extracts the specified ``fields`` from a numpy structured array
    and stacks them into a ``torch.Tensor``.

    Following TNC convention as a default, the returned tensor is of shape
    (time, field/batch, electrode_channel).

    Args:
        fields (list): List of field names to be extracted from the passed in
            structured numpy ndarray.
        stack_dim (int): The new dimension to insert while stacking
            ``fields``. (default: 1)
    """

    fields: Sequence[str] = ("emg_left", "emg_right")
    stack_dim: int = 1

    def __call__(self, data: np.ndarray) -> torch.Tensor:
        return torch.stack(
            [torch.as_tensor(data[f]) for f in self.fields], dim=self.stack_dim
        )


@dataclass
class Lambda:
    """Applies a custom lambda function as a transform.

    Args:
        lambd (lambda): Lambda to wrap within.
    """

    lambd: Transform[Any, Any]

    def __call__(self, data: Any) -> Any:
        return self.lambd(data)


@dataclass
class ForEach:
    """Applies the provided ``transform`` over each item of a batch
    independently. By default, assumes the input is of shape (T, N, ...).

    Args:
        transform (Callable): The transform to apply to each batch item of
            the input tensor.
        batch_dim (int): The bach dimension, i.e., the dim along which to
            unstack/unbind the input tensor prior to mapping over
            ``transform`` and restacking. (default: 1)
    """

    transform: Transform[torch.Tensor, torch.Tensor]
    batch_dim: int = 1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [self.transform(t) for t in tensor.unbind(self.batch_dim)],
            dim=self.batch_dim,
        )


@dataclass
class Compose:
    """Compose a chain of transforms.

    Args:
        transforms (list): List of transforms to compose.
    """

    transforms: Sequence[Transform[Any, Any]]

    def __call__(self, data: Any) -> Any:
        for transform in self.transforms:
            data = transform(data)
        return data


@dataclass
class RandomBandRotation:
    """Applies band rotation augmentation by shifting the electrode channels
    by an offset value randomly chosen from ``offsets``. By default, assumes
    the input is of shape (..., C).

    NOTE: If the input is 3D with batch dim (TNC), then this transform
    applies band rotation for all items in the batch with the same offset.
    To apply different rotations each batch item, use the ``ForEach`` wrapper.

    Args:
        offsets (list): List of integers denoting the offsets by which the
            electrodes are allowed to be shift. A random offset from this
            list is chosen for each application of the transform.
        channel_dim (int): The electrode channel dimension. (default: -1)
    """

    offsets: Sequence[int] = (-1, 0, 1)
    channel_dim: int = -1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        offset = np.random.choice(self.offsets) if len(self.offsets) > 0 else 0
        return tensor.roll(offset, dims=self.channel_dim)


@dataclass
class TemporalAlignmentJitter:
    """Applies a temporal jittering augmentation that randomly jitters the
    alignment of left and right EMG data by up to ``max_offset`` timesteps.
    The input must be of shape (T, ...).

    Args:
        max_offset (int): The maximum amount of alignment jittering in terms
            of number of timesteps.
        stack_dim (int): The dimension along which the left and right data
            are stacked. See ``ToTensor()``. (default: 1)
    """

    max_offset: int
    stack_dim: int = 1

    def __post_init__(self) -> None:
        assert self.max_offset >= 0

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.shape[self.stack_dim] == 2
        left, right = tensor.unbind(self.stack_dim)

        offset = np.random.randint(-self.max_offset, self.max_offset + 1)
        if offset > 0:
            left = left[offset:]
            right = right[:-offset]
        if offset < 0:
            left = left[:offset]
            right = right[-offset:]

        return torch.stack([left, right], dim=self.stack_dim)


@dataclass
class LogSpectrogram:
    """Creates log10-scaled spectrogram from an EMG signal. In the case of
    multi-channeled signal, the channels are treated independently.
    The input must be of shape (T, ...) and the returned spectrogram
    is of shape (T, ..., freq).

    Args:
        n_fft (int): Size of FFT, creates n_fft // 2 + 1 frequency bins.
            (default: 64)
        hop_length (int): Number of samples to stride between consecutive
            STFT windows. (default: 16)
    """

    n_fft: int = 64
    hop_length: int = 16

    def __post_init__(self) -> None:
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            normalized=True,
            # Disable centering of FFT windows to avoid padding inconsistencies
            # between train and test (due to differing window lengths), as well
            # as to be more faithful to real-time/streaming execution.
            center=False,
        )

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        x = tensor.movedim(0, -1)  # (T, ..., C) -> (..., C, T)
        spec = self.spectrogram(x)  # (..., C, freq, T)
        logspec = torch.log10(spec + 1e-6)  # Prevent log(0)
        return logspec.movedim(-1, 0)  # (T, ..., C, freq)


@dataclass
class SpecAugment:
    """Applies time and frequency masking as per the paper
    "SpecAugment: A Simple Data Augmentation Method for Automatic Speech
    Recognition, Park et al" (https://arxiv.org/abs/1904.08779).

    Args:
        n_time_masks (int): Maximum number of time masks to apply,
            uniformly sampled from 0. (default: 0)
        time_mask_param (int): Maximum length of each time mask,
            uniformly sampled from 0. (default: 0)
        iid_time_masks (int): Whether to apply different time masks to
            each band/channel (default: True)
        n_freq_masks (int): Maximum number of frequency masks to apply,
            uniformly sampled from 0. (default: 0)
        freq_mask_param (int): Maximum length of each frequency mask,
            uniformly sampled from 0. (default: 0)
        iid_freq_masks (int): Whether to apply different frequency masks to
            each band/channel (default: True)
        mask_value (float): Value to assign to the masked columns (default: 0.)
    """

    n_time_masks: int = 0
    time_mask_param: int = 0
    iid_time_masks: bool = True
    n_freq_masks: int = 0
    freq_mask_param: int = 0
    iid_freq_masks: bool = True
    mask_value: float = 0.0

    def __post_init__(self) -> None:
        self.time_mask = torchaudio.transforms.TimeMasking(
            self.time_mask_param, iid_masks=self.iid_time_masks
        )
        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            self.freq_mask_param, iid_masks=self.iid_freq_masks
        )

    def __call__(self, specgram: torch.Tensor) -> torch.Tensor:
        # (T, ..., C, freq) -> (..., C, freq, T)
        x = specgram.movedim(0, -1)

        # Time masks
        n_t_masks = np.random.randint(self.n_time_masks + 1)
        for _ in range(n_t_masks):
            x = self.time_mask(x, mask_value=self.mask_value)

        # Frequency masks
        n_f_masks = np.random.randint(self.n_freq_masks + 1)
        for _ in range(n_f_masks):
            x = self.freq_mask(x, mask_value=self.mask_value)

        # (..., C, freq, T) -> (T, ..., C, freq)
        return x.movedim(-1, 0)

@dataclass
class Augmentations:
    """
    Unified augmentation pipeline.
    When prob > 0, each augmentation in the list is applied with a 50% chance.
    """
    prob: float = 0.5  # Set to a nonzero value to enable augmentation.
    augmentations: Sequence[Transform[torch.Tensor, torch.Tensor]] = None

    def __post_init__(self):
        if self.augmentations is None:
            self.augmentations = [
                RandomBandRotation(offsets=(-1, 0, 1)),
                SpecAugment(n_time_masks=2, time_mask_param=5, n_freq_masks=2, freq_mask_param=5),
                TemporalAlignmentJitter(max_offset=3),
                self.time_warp,
                self.add_gaussian_noise,
                self.jitter,
                self.channel_dropout,
                self.frequency_shift,
            ]

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        if random.random() > self.prob:
            return signal
        for aug_func in self.augmentations:
            if random.random() > 0.5:
                signal = aug_func(signal)
        return signal
    
    @staticmethod    
    def time_warp(signal: torch.Tensor, alpha: float = 0.2) -> torch.Tensor:
        time_steps = np.linspace(0, 1, len(signal))
        stretched_steps = np.cumsum(np.random.uniform(1 - alpha, 1 + alpha, len(signal)))
        stretched_steps = stretched_steps / max(stretched_steps[-1], 1e-6)  # Avoid division by zero

        interpolator = interp1d(stretched_steps, signal.numpy(), axis=0, fill_value="extrapolate")
        warped_signal = interpolator(time_steps)

        return torch.tensor(np.nan_to_num(warped_signal, nan=0.0), dtype=torch.float32)  # Remove NaNs

    @staticmethod 
    def add_gaussian_noise(signal: torch.Tensor, noise_level: float = 0.01) -> torch.Tensor:
        noise = torch.tensor(np.random.normal(0, noise_level, signal.shape), dtype=torch.float32)
        return torch.nan_to_num(signal + noise, nan=0.0, posinf=1e6, neginf=-1e6)  # Ensure stability

    @staticmethod 
    def jitter(signal: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
        noise = torch.tensor(np.random.normal(0, sigma, signal.shape), dtype=torch.float32)
        return torch.nan_to_num(signal + noise, nan=0.0, posinf=1e6, neginf=-1e6)

    @staticmethod 
    def channel_dropout(signal: torch.Tensor, drop_rate: float = 0.1) -> torch.Tensor:
        mask = np.random.choice([0, 1], size=signal.shape, p=[drop_rate, 1 - drop_rate])
        dropped_signal = signal * torch.tensor(mask, dtype=torch.float32)
        
        return torch.nan_to_num(dropped_signal, nan=0.0, posinf=1e6, neginf=-1e6)  # Ensure stability

    @staticmethod 
    def frequency_shift(signal: torch.Tensor, shift_factor: float = 0.1) -> torch.Tensor:
        freq_signal = fft(signal.numpy())

        # Ensure valid shift
        shift_amount = int(shift_factor * len(freq_signal))
        shifted_signal = np.roll(freq_signal, shift_amount)

        # Convert back to real space and sanitize values
        result = torch.tensor(np.real(ifft(shifted_signal)), dtype=torch.float32)
        
        return torch.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)  # Remove NaNs and Infs
