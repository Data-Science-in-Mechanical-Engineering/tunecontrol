"""Random streams independent of Torch's global generator."""
import torch


def new_generator() -> torch.Generator:
    """Create a CPU generator seeded from system entropy, not global Torch state."""
    generator = torch.Generator(device="cpu")
    generator.seed()
    return generator
