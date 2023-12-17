import os

import numpy as np
import pandas as pd
import scipy
import torch
import torchaudio
from audiocraft.models import MusicGen
from tqdm import tqdm
from transformers import AutoProcessor, MusicgenForConditionalGeneration

from audio import audio_write
from config import CFG
from utils import set_seed


def inference_audiocraft(
    model, filenames, descriptions, output_dir, audio_path=None, continuation=False
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if continuation:
        if not audio_path:
            raise ValueError("audio_path must be provided for continuation inference")
        model.set_generation_params(
            duration=CFG.DURATION, temperature=0.9, continuation=True
        )
        x, sr = torchaudio.load(audio_path)
        wav, tokens = model.generate_continuation(
            prompt=x,
            prompt_sample_rate=sr,
            descriptions=descriptions,
            progress=True,
            return_tokens=True,
        )
    else:
        wav = model.generate(descriptions=descriptions, progress=True)
    for idx, one_wav in enumerate(wav):
        audio_write(
            stem_name=os.path.join(output_dir, os.path.splitext(filenames[idx])[0]),
            wav=one_wav.cpu(),
            sample_rate=CFG.SAMPLE_RATE,
            strategy=CFG.STRATEGY,
            loudness_compressor=CFG.LOUDNESS_COMPRESSOR,
            format=CFG.FORMAT,
            mp3_rate=320,
            peak_clip_headroom_db=1,
        )


def inference_huggingface(model, processor, filenames, descriptions, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    inputs = processor(
        text=descriptions,
        padding=True,
        return_tensors="pt",
    )

    audio_values = model.generate(
        **inputs.to(CFG.DEVICE), max_new_tokens=CFG.MAX_NEW_TOKENS
    )
    for idx, one_wav in enumerate(audio_values):
        output_path = os.path.join(output_dir, filenames[idx])
        torchaudio.save(output_path, one_wav[0], CFG.SAMPLE_RATE)
    # scipy.io.wavfile.write(output_path, rate=CFG.SAMPLE_RATE, data=audio_values[0, 0].numpy())


def inference_df(
    json_path,
    output_dir="output",
    model_type="audiocraft",
    continuation=False,
    model_path=None,
    audio_path=None,
    batch_size=1,
):
    df_inference = pd.read_json(json_path, orient="index").reset_index()
    df_inference.columns = ["filename", "description"]

    set_seed(CFG.SEED)
    if model_type == "audiocraft":
        model = MusicGen.get_pretrained(name=CFG.AUDIO_CRAFT_MODEL, device=CFG.DEVICE)
        if model_path:
            model.lm.load_state_dict(torch.load(model_path, map_location=CFG.DEVICE))
        model.set_generation_params(
            duration=5,
            temperature=1,
        )
    elif model_type == "huggingface":
        processor = AutoProcessor.from_pretrained(CFG.AUDIO_CRAFT_MODEL)
        model = MusicgenForConditionalGeneration.from_pretrained(CFG.AUDIO_CRAFT_MODEL)
        model.to(CFG.DEVICE)
    else:
        raise ValueError(f"model_type {model_type} not recognized")

    # for idx, row in tqdm(df_inference.iterrows()):
    for idx, batch in tqdm(
        df_inference.groupby(np.arange(len(df_inference)) // batch_size)
    ):  # tqdm(df_inference.iterrows()):
        print("Processing batch", batch)
        print("---------------------")
        filenames = batch["filename"].tolist()
        descriptions = batch["description"].tolist()
        if model_type == "audiocraft":
            inference_audiocraft(
                model=model,
                filenames=filenames,
                descriptions=descriptions,
                output_dir=os.path.join(output_dir, model_type),
                audio_path=audio_path,
                continuation=continuation,
            )
        elif model_type == "huggingface":
            inference_huggingface(
                model=model,
                processor=processor,
                filenames=filenames,
                descriptions=descriptions,
                output_dir=os.path.join(output_dir, model_type),
            )
        else:
            raise ValueError(f"model_type {model_type} not recognized")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="data/test/public.json")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument("--model_type", type=str, default="audiocraft")
    parser.add_argument("--continuation", action="store_true")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--audio_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    inference_df(
        json_path=args.json_path,
        output_dir=os.path.join(args.output_dir, args.version),
        model_type=args.model_type,
        continuation=args.continuation,
        model_path=args.model_path,
        audio_path=args.audio_path,
        batch_size=args.batch_size,
    )
