from itertools import pairwise

from jaxtyping import Float
from torch import Tensor
from torch import nn


class Classifier(nn.Module):
    """A stand-in model: by default an MLP of 88M parameters, about 353 MB in float32.

    Big enough that one copy per worker hurts. The server never imports it: it's exported to ONNX at build time.
    """

    def __init__(self, d_in: int = 1024, d: int = 4096, depth: int = 6, n_classes: int = 10) -> None:
        super().__init__()
        self.d_in = d_in
        widths = [d_in] + [d] * depth
        hidden = [layer for w_in, w_out in pairwise(widths) for layer in (nn.Linear(w_in, w_out), nn.ReLU())]
        self.net = nn.Sequential(*hidden, nn.Linear(widths[-1], n_classes))

    def forward(self, x: Float[Tensor, "batch d_in"]) -> Float[Tensor, "batch n_classes"]:
        logits: Tensor = self.net(x)
        return logits
