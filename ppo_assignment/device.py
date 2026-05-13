import torch


def resolve_device(requested_device, *, strict=False):
    requested = torch.device(requested_device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        message = (
            f"Requested CUDA device '{requested_device}', but this PyTorch build cannot use CUDA "
            f"(torch={torch.__version__}). Falling back to CPU."
        )
        if strict:
            raise RuntimeError(message)
        print(f"WARNING: {message}")
        return torch.device("cpu")
    return requested
