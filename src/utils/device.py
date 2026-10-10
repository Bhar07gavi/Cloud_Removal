import torch

def get_device(device: str = "auto") -> torch.device:
    if device in ("auto", "cuda"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "cpu":
        return torch.device("cpu")
    else:
        raise ValueError(f"Invalid device '{device}'. Valid options are 'auto', 'cuda', 'cpu'.")
