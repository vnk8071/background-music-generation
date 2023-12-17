import os
import random

import torch
import torch.nn as nn
import torchaudio
import wandb
from audiocraft.models import MusicGen
from audiocraft.modules.conditioners import ClassifierFreeGuidanceDropout
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_scheduler

from config import CFG
from utils import set_seed


class AudioDataset(Dataset):
    def __init__(self, data_dir, no_label=False):
        self.data_dir = data_dir
        self.data_map = []

        dir_map = os.listdir(data_dir)
        for d in dir_map:
            name, ext = os.path.splitext(d)
            if ext == ".mp3":
                if no_label:
                    self.data_map.append({"audio": os.path.join(data_dir, d)})
                    continue
                if os.path.exists(os.path.join(data_dir, name + ".txt")):
                    self.data_map.append(
                        {
                            "audio": os.path.join(data_dir, d),
                            "label": os.path.join(data_dir, name + ".txt"),
                        }
                    )
                else:
                    raise ValueError(f"No label file for {name}")

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx):
        data = self.data_map[idx]
        audio = data["audio"]
        label = data.get("label", "")

        return audio, label


def count_nans(tensor):
    nan_mask = torch.isnan(tensor)
    num_nans = torch.sum(nan_mask).item()
    return num_nans


def preprocess_audio(audio_path, model: MusicGen, duration: int = 10):
    wav, sr = torchaudio.load(audio_path)
    wav = torchaudio.functional.resample(wav, sr, model.sample_rate)
    wav = wav.mean(dim=0, keepdim=True)
    if wav.shape[1] < model.sample_rate * duration:
        return None
    end_sample = int(model.sample_rate * duration)
    # start_sample = random.randrange(0, max(wav.shape[1] - end_sample, 1))
    # wav = wav[:, start_sample: start_sample + end_sample]
    wav = torch.nn.functional.pad(wav, (0, end_sample - wav.shape[-1]))
    assert wav.shape[0] == 1

    wav = wav.cuda()
    wav = wav.unsqueeze(1)

    with torch.no_grad():
        gen_audio = model.compression_model.encode(wav)

    codes, scale = gen_audio
    assert scale is None

    return codes


def fixnan(tensor: torch.Tensor):
    nan_mask = torch.isnan(tensor)
    result = torch.where(nan_mask, torch.zeros_like(tensor), tensor)
    return result


def one_hot_encode(tensor, num_classes=2048):
    shape = tensor.shape
    one_hot = torch.zeros((shape[0], shape[1], num_classes))

    for i in range(shape[0]):
        for j in range(shape[1]):
            index = tensor[i, j].item()
            one_hot[i, j, index] = 1
    return one_hot


def train(
    dataset_path: str,
    model_id: str,
    lr: float,
    epochs: int,
    use_wandb: bool,
    no_label: bool = False,
    tune_text: bool = False,
    eval_step: int = 1000,
    save_step: int = 1000,
    grad_acc: int = 8,
    use_scaler: bool = False,
    weight_decay: float = 1e-5,
    warmup_steps: int = 10,
    batch_size: int = 10,
    use_cfg: bool = False,
):
    if use_wandb:
        run = wandb.init(project="audiocraft")

    set_seed(CFG.SEED)
    model = MusicGen.get_pretrained(model_id, CFG.DEVICE)
    model.lm = model.lm.to(torch.float32)  # important
    dataset_train_path = os.path.join(dataset_path, "train")
    dataset_train = AudioDataset(dataset_train_path, no_label=no_label)
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataset_val_path = os.path.join(dataset_path, "eval")
    dataset_val = AudioDataset(dataset_val_path, no_label=no_label)
    eval_dataloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    learning_rate = lr
    model.lm.train()

    scaler = torch.cuda.amp.GradScaler()

    if tune_text:
        print("Tuning text")
    else:
        print("Tuning everything")

    # from paper
    optimizer = AdamW(
        model.lm.condition_provider.parameters()
        if tune_text
        else model.lm.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
    )
    scheduler = get_scheduler(
        "cosine",
        optimizer,
        warmup_steps,
        int(epochs * len(train_dataloader) / grad_acc),
    )

    criterion = nn.CrossEntropyLoss()

    num_epochs = epochs
    save_models = False if save_step is None else True

    save_path = "models/"

    os.makedirs(save_path, exist_ok=True)

    current_step = 0

    for epoch in range(num_epochs):
        for batch_idx, (audio, label) in tqdm(enumerate(train_dataloader)):
            optimizer.zero_grad()

            all_codes = []
            texts = []

            for inner_audio, l in zip(audio, label):
                inner_audio = preprocess_audio(inner_audio, model, duration=10)
                if inner_audio is None:
                    continue

                if use_cfg:
                    codes = torch.cat([inner_audio, inner_audio], dim=0)
                else:
                    codes = inner_audio

                all_codes.append(codes)
                texts.append(open(l, "r", encoding="utf-8").read().strip())

            attributes, _ = model._prepare_tokens_and_attributes(texts, None)
            conditions = attributes
            if use_cfg:
                null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
                conditions = conditions + null_conditions
            tokenized = model.lm.condition_provider.tokenize(conditions)
            cfg_conditions = model.lm.condition_provider(tokenized)
            condition_tensors = cfg_conditions

            if len(all_codes) == 0:
                continue

            codes = torch.cat(all_codes, dim=0)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                lm_output = model.lm.compute_predictions(
                    codes=codes, conditions=[], condition_tensors=condition_tensors
                )

                codes = codes[0]
                logits = lm_output.logits[0]
                mask = lm_output.mask[0]

                codes = one_hot_encode(codes, num_classes=2048)

                codes = codes.cuda()
                logits = logits.cuda()
                mask = mask.cuda()

                mask = mask.view(-1)
                masked_logits = logits.view(-1, 2048)[mask]
                masked_codes = codes.view(-1, 2048)[mask]

                loss = criterion(masked_logits, masked_codes)

            current_step += 1

            assert count_nans(masked_logits) == 0

            (scaler.scale(loss) if use_scaler else loss).backward()

            total_norm = 0
            for p in model.lm.condition_provider.parameters():
                try:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                except AttributeError:
                    pass
            total_norm = total_norm ** (1.0 / 2)

            if use_wandb:
                run.log(
                    {
                        "loss": loss.item(),
                        "total_norm": total_norm,
                    }
                )

            if batch_idx % 500 == 0:
                print(
                    f"Epoch: {epoch}/{num_epochs} | Batch: {batch_idx}/{len(train_dataloader)} | Loss: {loss.item()}"
                )

            if batch_idx % grad_acc != grad_acc - 1:
                continue

            if use_scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.lm.parameters(), 0.5)

            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()

            if current_step % eval_step == 0:
                loss = torch.tensor(0.0).cuda()

                total_evals = 0

                with torch.no_grad():
                    for batch_idx, (audio_val, label_val) in tqdm(
                        enumerate(eval_dataloader)
                    ):
                        all_codes_val = []
                        texts_val = []

                        for inner_audio_val, l in zip(audio_val, label_val):
                            inner_audio_val = preprocess_audio(
                                inner_audio_val, model, duration=10
                            )
                            if inner_audio_val is None:
                                continue

                            if use_cfg:
                                codes = torch.cat([inner_audio, inner_audio], dim=0)
                            else:
                                codes = inner_audio

                            all_codes_val.append(codes)
                            texts_val.append(
                                open(l, "r", encoding="utf-8").read().strip()
                            )

                        attributes, _ = model._prepare_tokens_and_attributes(
                            texts_val, None
                        )
                        conditions = attributes

                        if use_cfg:
                            null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(
                                conditions
                            )
                            conditions = conditions + null_conditions
                        tokenized = model.lm.condition_provider.tokenize(conditions)
                        cfg_conditions = model.lm.condition_provider(tokenized)
                        condition_tensors = cfg_conditions

                        if len(all_codes_val) == 0:
                            continue

                        codes = torch.cat(all_codes, dim=0)

                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            lm_output = model.lm.compute_predictions(
                                codes=codes,
                                conditions=[],
                                condition_tensors=condition_tensors,
                            )

                            codes = codes[0]
                            logits = lm_output.logits[0]
                            mask = lm_output.mask[0]

                            codes = one_hot_encode(codes, num_classes=2048)

                            codes = codes.cuda()
                            logits = logits.cuda()
                            mask = mask.cuda()

                            mask = mask.view(-1)
                            masked_logits = logits.view(-1, 2048)[mask]
                            masked_codes = codes.view(-1, 2048)[mask]

                            loss_batch = criterion(masked_logits, masked_codes)
                            loss = loss + loss_batch

                            print(
                                f"Epoch: {epoch}/{num_epochs} | Batch: {batch_idx}/{len(eval_dataloader)} | Eval Loss: {loss_batch.item()}"
                            )
                            total_evals = total_evals + 1

                            if total_evals >= 10:
                                break

                    loss = loss / total_evals
                    print(f"Eval Loss: {loss.item()}")
                    run.log({"eval_loss": loss.item(), "epoch": epoch})

            if save_models:
                if (
                    current_step == int(current_step)
                    and int(current_step) % save_step == 0
                ):
                    torch.save(
                        model.lm.state_dict(), f"{save_path}/musicgen_{current_step}.pt"
                    )

    torch.save(model.lm.state_dict(), f"{save_path}/musicgen_final.pt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=False, default="small")
    parser.add_argument("--lr", type=float, required=False, default=1e-5)
    parser.add_argument("--epochs", type=int, required=False, default=10)
    parser.add_argument("--use_wandb", type=int, required=False, default=0)
    parser.add_argument("--eval_step", type=int, required=False, default=900)
    parser.add_argument("--save_step", type=int, required=False, default=900)
    parser.add_argument("--no_label", type=int, required=False, default=0)
    parser.add_argument("--tune_text", type=int, required=False, default=0)
    parser.add_argument("--weight_decay", type=float, required=False, default=1e-5)
    parser.add_argument("--grad_acc", type=int, required=False, default=2)
    parser.add_argument("--warmup_steps", type=int, required=False, default=10)
    parser.add_argument("--batch_size", type=int, required=False, default=10)
    parser.add_argument("--use_cfg", type=int, required=False, default=0)
    args = parser.parse_args()

    train(
        dataset_path=args.dataset_path,
        model_id=args.model_id,
        lr=args.lr,
        epochs=args.epochs,
        use_wandb=args.use_wandb,
        eval_step=args.eval_step,
        save_step=args.save_step,
        no_label=args.no_label,
        tune_text=args.tune_text,
        weight_decay=args.weight_decay,
        grad_acc=args.grad_acc,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        use_cfg=args.use_cfg,
    )
