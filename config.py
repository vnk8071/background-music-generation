import torch


class CFG:
    PROJECT = "ZAIC2023 - Background Music Generation"
    DURATION = 5
    TEMPERATURE = 1.0
    MAX_NEW_TOKENS = 256
    SAMPLE_RATE = 16000
    AUDIO_CRAFT_MODEL = "facebook/musicgen-small"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    STRATEGY = "loudness"
    LOUDNESS_COMPRESSOR = True
    FORMAT = "mp3"
    BITRATE = "320"
    PEAK = 1
    BATCH_SIZE = 1
    SEED = 8071
