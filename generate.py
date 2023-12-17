import argparse
import os
import typing as tp

import numpy as np
import pandas as pd
import torch
import torchaudio
from audiocraft.models import MultiBandDiffusion, MusicGen
from tqdm import tqdm

from audio import audio_write
from config import CFG
from utils import set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--json_path", type=str, required=True)
parser.add_argument("--batch_size", type=int, required=False, default=1)
parser.add_argument("--prompt", type=str, required=False)
parser.add_argument("--weights_path", type=str, required=False, default=None)
parser.add_argument("--model_id", type=str, required=False, default="small")
parser.add_argument("--save_path", type=str, required=False, default="output")
parser.add_argument("--multiband", type=bool, required=False, default=False)
parser.add_argument("--duration", type=float, required=False, default=5)
parser.add_argument("--sample_loops", type=int, required=False, default=1)
parser.add_argument("--use_sampling", type=bool, required=False, default=True)
parser.add_argument("--two_step_cfg", type=bool, required=False, default=False)
parser.add_argument("--top_k", type=int, required=False, default=250)
parser.add_argument("--top_p", type=float, required=False, default=0.0)
parser.add_argument("--temperature", type=float, required=False, default=1.0)
parser.add_argument("--cfg_coef", type=float, required=False, default=3.0)
args = parser.parse_args()


def inference(json_path, batch_size=CFG.BATCH_SIZE):
    output_dir = args.save_path
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    set_seed(CFG.SEED)
    if args.multiband:
        output_dir_diffusion = os.path.join(output_dir, "diffusion")
        if not os.path.exists(output_dir_diffusion):
            os.mkdir(output_dir_diffusion)
        mbd = MultiBandDiffusion.get_mbd_musicgen(device=CFG.DEVICE)

    model = MusicGen.get_pretrained(args.model_id, device=CFG.DEVICE)
    if args.weights_path is not None:
        model.lm.load_state_dict(torch.load(args.weights_path, map_location=CFG.DEVICE))
    duration = args.duration
    df_inference = pd.read_json(json_path, orient="index").reset_index()
    df_inference.columns = ["filename", "description"]

    for idx, batch in tqdm(
        df_inference.groupby(np.arange(len(df_inference)) // batch_size)
    ):  # tqdm(df_inference.iterrows()):
        print("Processing batch", batch)
        print("---------------------")
        filenames = batch["filename"].tolist()
        descriptions = batch["description"].tolist()

        attributes, prompt_tokens = model._prepare_tokens_and_attributes(
            descriptions, None
        )
        model.generation_params = {
            "max_gen_len": int(duration * model.frame_rate),
            "use_sampling": args.use_sampling,
            "temp": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "cfg_coef": args.cfg_coef,
            "two_step_cfg": args.two_step_cfg,
        }
        total = []
        for _ in range(args.sample_loops):
            with model.autocast:
                gen_tokens = model.lm.generate(
                    prompt_tokens, attributes, callback=None, **model.generation_params
                )
                total.append(
                    gen_tokens[
                        ...,
                        prompt_tokens.shape[-1] if prompt_tokens is not None else 0 :,
                    ]
                )
                prompt_tokens = gen_tokens[..., -gen_tokens.shape[-1] // 2 :]
        gen_tokens = torch.cat(total, -1)

        assert gen_tokens.dim() == 3
        with torch.no_grad():
            gen_audio = model.compression_model.decode(gen_tokens, None)

        if args.multiband:
            wav_diffusion = mbd.tokens_to_wav(gen_tokens)

        for idx, one_wav in enumerate(gen_audio):
            audio_write(
                stem_name=os.path.join(output_dir, os.path.splitext(filenames[idx])[0]),
                wav=one_wav.cpu(),
                sample_rate=CFG.SAMPLE_RATE,
                strategy=CFG.STRATEGY,
                loudness_compressor=CFG.LOUDNESS_COMPRESSOR,
                format=CFG.FORMAT,
                mp3_rate=CFG.BITRATE,
                peak_clip_headroom_db=CFG.PEAK,
            )
            if args.multiband:
                audio_write(
                    stem_name=os.path.join(
                        output_dir_diffusion, os.path.splitext(filenames[idx])[0]
                    ),
                    wav=wav_diffusion[idx].cpu(),
                    sample_rate=CFG.SAMPLE_RATE,
                    strategy=CFG.STRATEGY,
                    loudness_compressor=CFG.LOUDNESS_COMPRESSOR,
                    format=CFG.FORMAT,
                    mp3_rate=CFG.BITRATE,
                    peak_clip_headroom_db=CFG.PEAK,
                )


if __name__ == "__main__":
    inference(args.json_path, args.batch_size)
