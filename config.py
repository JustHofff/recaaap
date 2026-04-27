# config.py
import torch

DATASET_NAME = "jhofff/handwriting-lines-cleaned"
PT_MODEL_NAME = "microsoft/trocr-small-handwritten"
FT_MODEL_NAME = "jhofff/trocr-finetuned-handwriting"

OUTPUT_DIR = "output/parsed_lines"

SEED = 21

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"