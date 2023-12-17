import os

import pandas as pd
import torch
import torchaudio
from tqdm import tqdm


def create_loops(audio_file, loops=3):
    audio, sample_rate = torchaudio.load(audio_file)
    audio = torch.cat([audio] * loops, dim=1)
    torchaudio.save(
        os.path.join("data/train/gen30", os.path.basename(audio_file)),
        audio,
        sample_rate,
    )


if __name__ == "__main__":
    df = pd.read_json("data/train/train.json", orient="index").reset_index()
    df.columns = ["filename", "description"]
    for row in tqdm(df.iterrows()):
        create_loops(os.path.join("data/train/audio", row[1]["filename"]))
