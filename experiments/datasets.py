"""
Dataset loaders for multi-dataset scaling law experiments.
============================================================
Provides a unified interface for:
  - CIFAR-100 (32x32, 100 classes)
  - CIFAR-10  (32x32, 10 classes)
  - Tiny ImageNet (64x64, 200 classes)
  - Speech Commands v2 (mel-spectrogram 1x128x81, 35 classes)

Each get_*() function returns (train_dataset, test_dataset, num_classes, input_shape).
"""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

DATA_DIR = Path(__file__).parent / "data"


# =============================================================================
# Cutout augmentation (shared)
# =============================================================================

class Cutout:
    """Randomly mask out a square patch from the image."""
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        h, w = img.shape[1], img.shape[2]
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = max(0, y - self.size // 2)
        y2 = min(h, y + self.size // 2)
        x1 = max(0, x - self.size // 2)
        x2 = min(w, x + self.size // 2)
        img[:, y1:y2, x1:x2] = 0.0
        return img


# =============================================================================
# CIFAR-100
# =============================================================================

def get_cifar100(root=None):
    """CIFAR-100 with standard augmentation + Cutout."""
    root = str(root or DATA_DIR)
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        Cutout(8),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.CIFAR100(root=root, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR100(root=root, train=False, download=True, transform=test_transform)
    return train_set, test_set, 100, (3, 32, 32)


# =============================================================================
# CIFAR-10
# =============================================================================

def get_cifar10(root=None):
    """CIFAR-10 with standard augmentation + Cutout."""
    root = str(root or DATA_DIR)
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        Cutout(8),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.CIFAR10(root=root, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root=root, train=False, download=True, transform=test_transform)
    return train_set, test_set, 10, (3, 32, 32)


# =============================================================================
# Tiny ImageNet (64x64, 200 classes)
# =============================================================================

def get_tiny_imagenet(root=None):
    """
    Tiny ImageNet: 200 classes, 500 train + 50 val images per class, 64x64.
    Downloads and extracts if not present. Uses val set as test set.
    """
    root = Path(root or DATA_DIR)
    tinyimagenet_dir = root / "tiny-imagenet-200"

    if not tinyimagenet_dir.exists():
        _download_tiny_imagenet(root)

    mean = (0.4802, 0.4481, 0.3975)
    std = (0.2770, 0.2691, 0.2821)

    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        Cutout(16),  # Scaled cutout for 64x64
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.ImageFolder(str(tinyimagenet_dir / "train"), transform=train_transform)

    # Restructure val if needed (Tiny ImageNet val has flat layout by default)
    val_dir = tinyimagenet_dir / "val"
    _restructure_tiny_imagenet_val(val_dir)
    test_set = datasets.ImageFolder(str(val_dir), transform=test_transform)

    return train_set, test_set, 200, (3, 64, 64)


def _download_tiny_imagenet(root):
    """Download and extract Tiny ImageNet."""
    import urllib.request
    import zipfile

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = root / "tiny-imagenet-200.zip"

    if not zip_path.exists():
        print(f"Downloading Tiny ImageNet to {zip_path}...")
        urllib.request.urlretrieve(url, str(zip_path))
        print("Download complete.")

    print("Extracting Tiny ImageNet...")
    with zipfile.ZipFile(str(zip_path), "r") as z:
        z.extractall(str(root))
    print("Extraction complete.")


def _restructure_tiny_imagenet_val(val_dir):
    """
    Restructure Tiny ImageNet val/ into class subdirectories for ImageFolder.
    The original layout has all images in val/images/ with val_annotations.txt.
    """
    val_dir = Path(val_dir)
    annotations_file = val_dir / "val_annotations.txt"
    images_dir = val_dir / "images"

    # Already restructured if annotations file doesn't exist or images dir is gone
    if not annotations_file.exists() or not images_dir.exists():
        return

    print("Restructuring Tiny ImageNet val directory...")
    with open(annotations_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            img_name, class_id = parts[0], parts[1]
            class_dir = val_dir / class_id / "images"
            class_dir.mkdir(parents=True, exist_ok=True)
            src = images_dir / img_name
            dst = class_dir / img_name
            if src.exists():
                src.rename(dst)

    # Clean up flat images dir if empty
    if images_dir.exists() and not any(images_dir.iterdir()):
        images_dir.rmdir()
    print("Val restructuring complete.")


# =============================================================================
# Speech Commands v2 (mel-spectrogram, 35 classes)
# =============================================================================

class SpeechCommandsMel(Dataset):
    """
    Speech Commands V2 with mel-spectrogram preprocessing.
    Uses soundfile to load wav files directly (no torchaudio/torchcodec/FFmpeg).
    Output shape: (1, 128, 81) — 1 channel, 128 mel bins, ~81 time frames (1 sec @ 16kHz).
    """

    KEYWORDS = [
        "backward", "bed", "bird", "cat", "dog", "down", "eight",
        "five", "follow", "forward", "four", "go", "happy", "house",
        "learn", "left", "marvin", "nine", "no", "off", "on", "one",
        "right", "seven", "sheila", "six", "stop", "three", "tree",
        "two", "up", "visual", "wow", "yes", "zero",
    ]
    SAMPLE_RATE = 16000
    SC_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"

    def __init__(self, root, subset="training", augment=False):
        import soundfile as sf
        self._sf = sf
        self.root = Path(root) / "SpeechCommands" / "speech_commands_v0.02"
        self.augment = augment

        # Download if needed
        if not self.root.exists():
            self._download(root)

        # Build file list
        self._files = []
        self._labels = sorted(self.KEYWORDS)
        self._label_to_idx = {label: i for i, label in enumerate(self._labels)}

        # Load subset split lists
        val_list = self._load_list("validation_list.txt")
        test_list = self._load_list("testing_list.txt")

        for label in self._labels:
            label_dir = self.root / label
            if not label_dir.is_dir():
                continue
            for wav_path in sorted(label_dir.glob("*.wav")):
                rel = f"{label}/{wav_path.name}"
                if subset == "training" and rel not in val_list and rel not in test_list:
                    self._files.append((wav_path, label))
                elif subset == "validation" and rel in val_list:
                    self._files.append((wav_path, label))
                elif subset == "testing" and rel in test_list:
                    self._files.append((wav_path, label))

        # Mel filterbank (lazy init)
        self._n_fft = 1024
        self._hop_length = 200
        self._n_mels = 128
        self._mel_basis = None

    def _download(self, root):
        """Download and extract Speech Commands v0.02."""
        import tarfile, urllib.request
        tar_path = Path(root) / "speech_commands_v0.02.tar.gz"
        self.root.mkdir(parents=True, exist_ok=True)
        if not tar_path.exists():
            print(f"Downloading Speech Commands v2 to {tar_path}...")
            urllib.request.urlretrieve(self.SC_URL, str(tar_path))
        print(f"Extracting to {self.root}...")
        with tarfile.open(str(tar_path), "r:gz") as tar:
            tar.extractall(str(self.root))

    def _load_list(self, filename):
        path = self.root / filename
        if not path.exists():
            return set()
        with open(path) as f:
            return set(line.strip() for line in f if line.strip())

    def _get_mel_basis(self):
        if self._mel_basis is not None:
            return self._mel_basis
        n_freqs = self._n_fft // 2 + 1
        def hz_to_mel(hz): return 2595.0 * np.log10(1.0 + hz / 700.0)
        def mel_to_hz(mel): return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
        mel_points = np.linspace(hz_to_mel(0.0), hz_to_mel(8000.0), self._n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bin_points = np.floor((self._n_fft + 1) * hz_points / self.SAMPLE_RATE).astype(int)
        fbank = np.zeros((self._n_mels, n_freqs))
        for i in range(self._n_mels):
            for j in range(bin_points[i], bin_points[i + 1]):
                fbank[i, j] = (j - bin_points[i]) / max(1, bin_points[i + 1] - bin_points[i])
            for j in range(bin_points[i + 1], bin_points[i + 2]):
                fbank[i, j] = (bin_points[i + 2] - j) / max(1, bin_points[i + 2] - bin_points[i + 1])
        self._mel_basis = torch.FloatTensor(fbank)
        return self._mel_basis

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx):
        wav_path, label = self._files[idx]
        waveform, sr = self._sf.read(str(wav_path), dtype="float32")
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]
        else:
            waveform = waveform.T
        waveform = torch.FloatTensor(waveform)

        # Pad or trim to 1 second
        target_length = self.SAMPLE_RATE
        if waveform.shape[1] < target_length:
            waveform = torch.nn.functional.pad(waveform, (0, target_length - waveform.shape[1]))
        else:
            waveform = waveform[:, :target_length]

        # STFT -> mel spectrogram
        stft = torch.stft(waveform[0], n_fft=self._n_fft, hop_length=self._hop_length,
                          return_complex=True, window=torch.hann_window(self._n_fft))
        power = stft.abs() ** 2
        mel = self._get_mel_basis() @ power
        mel = mel.unsqueeze(0)  # (1, 128, 81)

        mel = torch.log(mel + 1e-9)
        mel = (mel - mel.mean()) / (mel.std() + 1e-9)

        if self.augment:
            mel = self._spec_augment(mel)

        label_idx = self._label_to_idx.get(label, 0)
        return mel, label_idx

    @staticmethod
    def _spec_augment(mel):
        """Simple SpecAugment: frequency and time masking."""
        _, n_mels, n_frames = mel.shape
        f_mask_width = np.random.randint(0, min(15, n_mels // 4) + 1)
        if f_mask_width > 0:
            f_start = np.random.randint(0, n_mels - f_mask_width)
            mel[:, f_start:f_start + f_mask_width, :] = 0.0
        t_mask_width = np.random.randint(0, min(10, n_frames // 4) + 1)
        if t_mask_width > 0:
            t_start = np.random.randint(0, n_frames - t_mask_width)
            mel[:, :, t_start:t_start + t_mask_width] = 0.0
        return mel


def get_speech_commands(root=None):
    """Speech Commands v2 with mel-spectrogram preprocessing."""
    root = str(root or DATA_DIR)
    train_set = SpeechCommandsMel(root, subset="training", augment=True)
    test_set = SpeechCommandsMel(root, subset="testing", augment=False)
    num_classes = len(train_set._labels)
    return train_set, test_set, num_classes, (1, 128, 81)


# =============================================================================
# Unified interface
# =============================================================================

DATASET_REGISTRY = {
    "cifar100": get_cifar100,
    "cifar10": get_cifar10,
    "tinyimagenet": get_tiny_imagenet,
    "speechcommands": get_speech_commands,
}


def get_dataset(name, root=None):
    """
    Get dataset by name. Returns (train_set, test_set, num_classes, input_shape).

    Args:
        name: One of 'cifar100', 'cifar10', 'tinyimagenet', 'speechcommands'
        root: Data root directory (default: experiments/data/)
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name](root)
