import os
import random

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing
import torch
from audiocraft.data.audio import audio_write
from audiocraft.models import MusicGen


def plot_samples_spectrum(data_dir, df, n_samples=3):
    df_sample = df.sample(n_samples)
    for i, row in df_sample.iterrows():
        audio_array, sample_rate = librosa.load(os.path.join(data_dir, row["filename"]))
        X = librosa.stft(y=audio_array)
        Xdb = librosa.amplitude_to_db(abs(X))
        plt.figure(figsize=(15, 6))
        plt.title(row["filename"])
        plt.annotate(
            text="Description:\n" + row["description"].replace(".", "\n"),
            xy=(0, 0),
            xytext=(0, -0.3),
            xycoords="axes fraction",
            fontsize=12,
        )
        librosa.display.specshow(Xdb, sr=sample_rate, x_axis="time", y_axis="log")


def plot_samples_wave(data_dir, df, n_samples=3):
    df_sample = df.sample(n_samples)
    for i, row in df_sample.iterrows():
        audio_array, sample_rate = librosa.load(os.path.join(data_dir, row["filename"]))
        plt.figure(figsize=(15, 6))
        plt.title(row["filename"])
        plt.annotate(
            text="Description:\n" + row["description"].replace(".", "\n"),
            xy=(0, 0),
            xytext=(0, -0.3),
            xycoords="axes fraction",
            fontsize=12,
        )
        librosa.display.waveshow(audio_array, sr=sample_rate, color="blue")


def plot_samples_mfcc(data_dir, df, n_samples=3):
    df_sample = df.sample(n_samples)
    for i, row in df_sample.iterrows():
        audio_array, sample_rate = librosa.load(os.path.join(data_dir, row["filename"]))
        mfccs = librosa.feature.mfcc(y=audio_array, sr=sample_rate)
        mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
        plt.figure(figsize=(15, 6))
        plt.title(row["filename"])
        plt.annotate(
            text="Description:\n" + row["description"].replace(".", "\n"),
            xy=(0, 0),
            xytext=(0, -0.3),
            xycoords="axes fraction",
            fontsize=12,
        )
        librosa.display.specshow(mfccs, sr=sample_rate, x_axis="time", y_axis="mel")


def inference_text(descriptions: list[str]):
    model = MusicGen.get_pretrained("facebook/musicgen-small")
    model.set_generation_params(duration=10)

    if descriptions:
        wav = model.generate(descriptions, progress=True)  # generates 3 samples.

        for idx, one_wav in enumerate(wav):
            # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
            audio_write(
                f"{idx}",
                one_wav.cpu(),
                model.sample_rate,
                strategy="loudness",
                loudness_compressor=True,
            )
    else:
        raise ValueError("descriptions must be provided")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
