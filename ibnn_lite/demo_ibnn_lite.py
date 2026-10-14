"""Demo: what IBNN Lite does that a convolution does not.

Run with:   python3 demo_ibnn_lite.py

The stimulus is a luminance step. With lambda = 0 the layer is exactly a
convolution and reproduces the step unchanged. With lambda > 0 the correction
switches on and the two pixels flanking the step are pushed apart: the bright
side is enhanced and the dark side is suppressed, while uniform regions are
left untouched. That is edge enhancement, the same asymmetry around a step that
produces Mach bands in human vision.
"""

import torch
import torch.nn.functional as F

from ibnn_lite import IBNNLite


def step_image(size=9, edge=5):
    """A single-channel image, dark on the left and bright on the right.

    Inputs
      size  int. Side of the square image.
      edge  int. Column at which the step begins.

    Outputs
      Tensor (1, 1, size, size).
    """
    x = torch.zeros(1, 1, size, size)
    x[:, :, :, edge:] = 1.0
    return x


def identity_layer(lam):
    """An IBNNLite whose M is a 1x1 identity, so U is the input itself.

    This isolates the correction: any difference from the input must come from
    the IBNN term and not from the convolution.

    Inputs
      lam  float. Weight of the correction.

    Outputs
      IBNNLite with one input and one output channel.
    """
    layer = IBNNLite(1, 1, m_kernel_size=1, w_kernel_size=3,
                     phi=lambda z: z, lambda_init=lam)
    with torch.no_grad():
        layer.m.fill_(1.0)
        layer.b.zero_()
    return layer


def main():
    x = step_image()
    row = 4                                  # central row, away from the border

    print("input (central row)")
    print("   ", [round(v, 2) for v in x[0, 0, row].tolist()])

    for lam in (0.0, 1.0):
        y = identity_layer(lam)(x)
        print(f"\noutput with lambda = {lam}")
        print("   ", [round(v, 2) for v in y[0, 0, row].tolist()])

    y0 = identity_layer(0.0)(x)
    y1 = identity_layer(1.0)(x)
    d = (y1 - y0)[0, 0, row]
    print("\ncorrection applied (output with lambda=1 minus the convolution)")
    print("   ", [round(v, 2) for v in d.tolist()])
    print("\n    zero everywhere except at the step, where it is antisymmetric:")
    print(f"    dark side  {d[4]:+.2f}      bright side {d[5]:+.2f}")

    # The same layer used normally, as one layer among others.
    layer = IBNNLite(3, 16, m_kernel_size=3, w_kernel_size=3, lambda_init=0.5)
    out = layer(torch.randn(8, 3, 32, 32))
    print(f"\nas an ordinary layer: (8, 3, 32, 32) -> {tuple(out.shape)}")
    n_train = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in layer.parameters())
    print(f"parameters: {n_total} in total, {n_train} trainable "
          f"(Omega is frozen by default)")


if __name__ == "__main__":
    main()
