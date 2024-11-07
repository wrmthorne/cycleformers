import zlib

import torch


def encode_name_for_forward(*texts: str, return_tensors: bool = False) -> torch.Tensor | list[int] | int:
    """Encodes dataset names and adapter names to 32-bit integers. Returns a single int if one text is passed,
    a list of ints if multiple texts are passed, or a torch.int32 tensor if return_tensors is True."""
    if not texts:
        if return_tensors:
            return torch.tensor([], dtype=torch.int32)
        return []

    # Mask to 31 bits to ensure values fit in int32
    hash_values = [zlib.crc32(str(text).encode()) & 0x7FFFFFFF for text in texts]

    if return_tensors:
        return torch.tensor(hash_values, dtype=torch.int32)

    if len(hash_values) == 1:
        return hash_values[0]

    return hash_values
