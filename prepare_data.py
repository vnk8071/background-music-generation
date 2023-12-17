import json
import os
import random
import shutil

import librosa
import pandas as pd
from tqdm import tqdm

from config import CFG


def prepare_dataset(format: str = "txt"):
    df_inference = pd.read_json("data/train/train.json", orient="index").reset_index()
    df_inference.columns = ["filename", "description"]
    if format == "txt":
        if not os.path.exists("data/train/dataset_train_val"):
            os.makedirs("data/train/dataset_train_val", exist_ok=True)

        for row in df_inference.iterrows():
            shutil.copy(
                f"data/train/audio/{row[1]['filename']}",
                f"data/train/dataset_train_val/{row[1]['filename']}",
            )
            with open(
                f"data/train/dataset_train_val/{row[1]['filename'][:-4]}.txt", "w"
            ) as f:
                f.write(row[1]["description"])

    elif format == "json":
        os.makedirs("egs/train", exist_ok=True)
        os.makedirs("egs/eval", exist_ok=True)
        dataset_path = "dataset/audio"

        train_len = 0
        eval_len = 0

        for row in df_inference.iterrows():
            shutil.copy(
                f"data/train/audio/{row[1]['filename']}",
                f"dataset/audio/{row[1]['filename']}",
            )

        with open("egs/train/data.jsonl", "w") as train_file, open(
            "egs/eval/data.jsonl", "w"
        ) as eval_file:
            for i, row in tqdm(df_inference.iterrows()):
                # y, sr = librosa.load(os.path.join(dataset_path, row['filename']))
                # length = librosa.get_duration(y=y, sr=sr)
                entry = {
                    "key": row["filename"],
                    "artist": "",
                    "sample_rate": CFG.SAMPLE_RATE,
                    "file_extension": "mp3",
                    "description": row["description"],
                    "keywords": "",
                    "duration": 10,
                    "bpm": "",
                    "genre": "",
                    "title": "",
                    "name": "",
                    "instrument": "",
                    "moods": [],
                    "path": os.path.join(dataset_path, row["filename"]),
                }
                with open(
                    os.path.join(dataset_path, row["filename"][:-4] + ".json"), "w"
                ) as f:
                    f.write(json.dumps(entry))
                if random.random() < 0.9:
                    train_len += 1
                    train_file.write(json.dumps(entry) + "\n")
                else:
                    eval_len += 1
                    eval_file.write(json.dumps(entry) + "\n")

        print(train_len)
        print(eval_len)


if __name__ == "__main__":
    prepare_dataset(format="json")
