import os
import tqdm
import glob
import numpy as np
import torch
import json
import pandas as pd
from torch.utils.data import Dataset
from typing import Tuple, Optional
from fsd50_src.src.data.audio_parser import AudioParser
from fsd50_src.src.data.utils import load_audio


class SpectrogramDataset(Dataset):
    def __init__(self, manifest_path: str, labels_map: str,
                 audio_config: dict, mode: Optional[str] = "multilabel",
                 augment: Optional[bool] = False,
                 labels_delimiter: Optional[str] = ",",
                 mixer: Optional = None,
                 transform: Optional = None) -> None:
        super(SpectrogramDataset, self).__init__()
        assert os.path.isfile(labels_map)
        assert os.path.splitext(labels_map)[-1] == ".json"
        assert audio_config is not None
        with open(labels_map, 'r') as fd:
            self.labels_map = json.load(fd)

        lab = ['Domestic_sounds_and_home_sounds',
                'Vehicle',
                'Animal',
                'Human_voice',
                'Water',
                'Mechanisms',
                'Alarm',
                'Human_group_actions',
                'Explosion',
                'Engine'
               ]
        self.labels_map = {k: v for v, k in enumerate(lab)}
        self.len = None
        self.labels_delim = labels_delimiter
        df = pd.read_csv(manifest_path)
        self.files = df['files'].values
        self.labels = df['labels'].values
        if mode == 'multiclass':
            self.labels = self.multiclass_from_multilabel()
            self.labels = self.sym_noise_multiclass()
        print(self.labels[0])
        assert len(self.files) == len(self.labels)
        self.len = len(self.files)
        self.sr = audio_config.get("sample_rate", "22050")
        self.n_fft = audio_config.get("n_fft", 511)
        win_len = audio_config.get("win_len", None)
        if not win_len:
            self.win_len = self.n_fft
        else:
            self.win_len = win_len
        hop_len = audio_config.get("hop_len", None)
        if not hop_len:
            self.hop_len = self.n_fft // 2
        else:
            self.hop_len = hop_len

        self.normalize = audio_config.get("normalize", True)
        self.augment = augment
        self.min_duration = audio_config.get("min_duration", None)
        self.background_noise_path = audio_config.get("bg_files", None)
        if self.background_noise_path is not None:
            if os.path.exists(self.background_noise_path):
                self.bg_files = glob.glob(os.path.join(self.background_noise_path, "*.wav"))
        else:
            self.bg_files = None

        self.mode = mode
        feature = audio_config.get("feature", "spectrogram")
        self.spec_parser = AudioParser(n_fft=self.n_fft, win_length=self.win_len,
                                       hop_length=self.hop_len, feature=feature)
        self.mixer = mixer
        self.transform = transform

        if self.bg_files is not None:
            print("prepping bg_features")
            self.bg_features = []
            for f in tqdm.tqdm(self.bg_files):
                preprocessed_audio = self.__get_audio__(f)
                real, comp = self.__get_feature__(preprocessed_audio)
                self.bg_features.append(real)
        else:
            self.bg_features = None
        self.prefetched_labels = None
        if self.mode == "multilabel":
            self.fetch_labels()

    def fetch_labels(self):
        self.prefetched_labels = []
        for lbl in tqdm.tqdm(self.labels):
            proc_lbl = self.__parse_labels__(lbl)
            self.prefetched_labels.append(proc_lbl.unsqueeze(0))
        self.prefetched_labels = torch.cat(self.prefetched_labels, dim=0)
        print(self.prefetched_labels.shape)

    def __get_audio__(self, f):
        audio = load_audio(f, self.sr, self.min_duration)
        return audio

    def __get_feature__(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        real, comp = self.spec_parser(audio)
        return real, comp

    def get_bg_feature(self, index: int) -> torch.Tensor:
        if self.bg_files is None:
            return None
        real = self.bg_features[index]
        if self.transform is not None:
            real = self.transform(real)
        return real

    def __get_item_helper__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f = self.files[index]
        lbls = self.labels[index]
        if self.mode == 'multilabel':
            label_tensor = self.__parse_labels__(lbls)
        else:
            label_tensor = lbls
        preprocessed_audio = self.__get_audio__(f)
        real, comp = self.__get_feature__(preprocessed_audio)
        if self.transform is not None:
            real = self.transform(real)
        return real, comp, label_tensor

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        real, comp, label_tensor = self.__get_item_helper__(index)
        if self.mixer is not None:
            real, final_label = self.mixer(self, real, label_tensor)
            if self.mode != "multiclass":

                return real, final_label

        return real, label_tensor

    def __parse_labels__(self, lbls: str) -> torch.Tensor:
        if self.mode == "multilabel":
            label_tensor = torch.zeros(len(self.labels_map)).float()
            for lbl in lbls.split(self.labels_delim):
                label_tensor[self.labels_map[lbl]] = 1

            return label_tensor
        elif self.mode == "multiclass":
            # take the first label, HOTFIX
            # label_tensor = torch.zeros(len(self.labels_map)).float()
            # label_tensor[self.labels_map[lbls.split(self.labels_delim)[0]]] = 1
            return self.labels_map[lbls.split(self.labels_delim)[0]]

    def __len__(self):
        return self.len

    def get_bg_len(self):
        return len(self.bg_files)

    def multiclass_from_multilabel(self):
        print('Getting single label for samples ...')
        labels = [self.labels_map[lbl.split(self.labels_delim)[0]] for lbl in self.labels]
        return np.array(labels)

    def sym_noise_multiclass(self, noise_rate=0.4):
        np.random.seed(42)
        print(f'Adding symmetric label noise with {noise_rate} rate')
        min_class = np.min(self.labels)
        max_class = np.max(self.labels)
        noisy_labels = []
        for label in tqdm.tqdm(self.labels):
            if np.random.uniform() < noise_rate:
                new_label = np.random.randint(low=min_class, high=max_class + 1)
                while new_label == label:
                    new_label = np.random.randint(low=min_class, high=max_class + 1)
                noisy_labels.append(new_label)
            else:
                noisy_labels.append(label)
        return np.array(noisy_labels)
