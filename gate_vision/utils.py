import torch
import yaml
import numpy as np
from PIL import Image


def collate(batch):
    imgs    = torch.stack([b[0] for b in batch])
    targets = torch.stack([b[1] for b in batch])
    return imgs, targets


# load the yaml file
def gather_info(file_path: str) -> dict:
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

# 
def pick_hardware() -> torch.device:
    return (torch.device('cuda') if torch.cuda.is_available() # check if cuda (gpu) is available
          else torch.device('mps') if torch.backends.mps.is_available() # check if metal performance (mac gpu [m series]) is available
          else torch.device('cpu')) # check if cpu is available


def preprocess_image(pil_img: Image.Image, size: int) -> torch.Tensor:
    gray = pil_img.convert("L").resize((size, size), Image.BILINEAR)
    arr  = np.array(gray, dtype=np.float32) / 255.0
    return torch.from_numpy(arr[np.newaxis, np.newaxis]).float()
