import torch.utils.data
import torchaudio
import os
from utils import *
import random
from natsort import natsorted

import numpy as np  # 変更点
import wave # 変更店

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from torch.utils.data.distributed import DistributedSampler


class DemandDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, cut_len=16000 * 2):
        self.cut_len = cut_len
        self.clean_dir = os.path.join(data_dir, "clean")
        self.noisy_dir = os.path.join(data_dir, "noisy")
        self.clean_wav_name = os.listdir(self.clean_dir)
        self.clean_wav_name = natsorted(self.clean_wav_name)

    def __len__(self):
        return len(self.clean_wav_name)

    def __getitem__(self, idx):
        clean_file = os.path.join(self.clean_dir, self.clean_wav_name[idx])
        noisy_file = os.path.join(self.noisy_dir, self.clean_wav_name[idx])
        # print(f'clean_file:{clean_file}')
        # print(f'noise_file:{noisy_file}')
        with wave.open(clean_file, "r") as wav: # clean_fileの読み込み(numpy)
            prm = wav.getparams()  # パラメータオブジェクト
            clean_data = wav.readframes(wav.getnframes())  # 音声データの読み込み(バイナリ形式)
            clean_data = np.frombuffer(clean_data, dtype=np.int16)  # 振幅に変換
            # wave_data = wave_data / np.iinfo(np.int16).max  # 最大値で正規化
            clean_data = clean_data.astype(np.float64)
        with wave.open(clean_file, "r") as wav: # noisy_fileの読み込み(numpy)
            prm = wav.getparams()  # パラメータオブジェクト
            noisy_data = wav.readframes(wav.getnframes())  # 音声データの読み込み(バイナリ形式)
            noisy_data = np.frombuffer(noisy_data, dtype=np.int16)  # 振幅に変換
            # wave_data = wave_data / np.iinfo(np.int16).max  # 最大値で正規化
            noisy_data = noisy_data.astype(np.float64)
        clean_ds = torch.from_numpy(clean_data.astype(np.float32))
        noisy_ds = torch.from_numpy(noisy_data.astype(np.float32))
        # print(f'np.clean:{clean_data}')
        # print(f'np.noise:{noisy_data}')

        # clean_ds, _ = torchaudio.load(clean_file, format="wav")
        # noisy_ds, _ = torchaudio.load(noisy_file, format="wav")
        # print(f'tensor.clean:{clean_ds}')
        # print(f'tensor.noise:{noisy_ds}')
        
        clean_ds = clean_ds.squeeze()
        noisy_ds = noisy_ds.squeeze()
        length = len(clean_ds)
        assert length == len(noisy_ds)
        if length < self.cut_len:
            units = self.cut_len // length
            clean_ds_final = []
            noisy_ds_final = []
            for i in range(units):
                clean_ds_final.append(clean_ds)
                noisy_ds_final.append(noisy_ds)
            clean_ds_final.append(clean_ds[: self.cut_len % length])
            noisy_ds_final.append(noisy_ds[: self.cut_len % length])
            clean_ds = torch.cat(clean_ds_final, dim=-1)
            noisy_ds = torch.cat(noisy_ds_final, dim=-1)
        else:
            # randomly cut 2 seconds segment
            wav_start = random.randint(0, length - self.cut_len)
            noisy_ds = noisy_ds[wav_start : wav_start + self.cut_len]
            clean_ds = clean_ds[wav_start : wav_start + self.cut_len]

        return clean_ds, noisy_ds, length


def load_data(ds_dir, batch_size, n_cpu, cut_len):
    torchaudio.set_audio_backend("sox_io")  # in linux
    train_dir = os.path.join(ds_dir, "train")
    test_dir = os.path.join(ds_dir, "test")

    train_ds = DemandDataset(train_dir, cut_len)
    test_ds = DemandDataset(test_dir, cut_len)

    train_dataset = torch.utils.data.DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(train_ds),
        drop_last=True,
        num_workers=n_cpu,
    )
    test_dataset = torch.utils.data.DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(test_ds),
        drop_last=False,
        num_workers=n_cpu,
    )

    return train_dataset, test_dataset
