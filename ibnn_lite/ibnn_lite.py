"""IBNN Lite: a compact, readable implementation of the layer.

This is a self-contained reimplementation, written so that the whole operation
can be read top to bottom. It is numerically equivalent to the IBNNLiteLayer of
the full library (agreement to 1.5e-05 across shapes, batch sizes, channel
counts and kernel sizes; see demo_ibnn_lite.py). Use the full library for
production work: it adds a fast approximate mode, many initialisations and the
rest of the INRF family.

The layer computes

    U = conv2d(I, M) - b
    Y = Phi( U - lambda * ( Omega * sigma( U(neighbour) - U(centre) ) ) )

The first line is an ordinary convolution. The second subtracts a correction
that measures, at every pixel, how much that pixel differs from its neighbours.
The inner nonlinearity sigma is what makes the layer irreducible: without it the
correction would be linear and the whole layer would collapse into a single
convolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IBNNLite(nn.Module):
    """IBNN Lite layer.

    Inputs
      in_channels       int. Channels of the input, e.g. 3 for an RGB image.
      out_channels      int. Channels of the output, i.e. how many filters M.
      m_kernel_size     int. Side of the M filter. 3 gives a 3x3 window.
      w_kernel_size     int. Side of the neighbourhood that Omega pools over.
      phi               callable. Outer nonlinearity. Default: ReLU.
      sigma             callable. Inner nonlinearity, applied to the differences
                        with the neighbours. Default: tanh.
      sigma_x_compress  float. Scales the difference before sigma. The larger it
                        is, the sooner sigma saturates. Default 10.0, which
                        makes the layer sensitive to differences below ~0.3 and
                        blind to anything larger.
      lambda_init       float. Initial weight of the correction. With 0 the
                        layer is exactly a convolution followed by phi.

    Outputs
      Tensor (B, out_channels, H, W), the same height and width as the input.

    Example
      layer = IBNNLite(3, 16, m_kernel_size=3, w_kernel_size=3)
      y = layer(torch.randn(8, 3, 32, 32))      # -> (8, 16, 32, 32)
    """

    def __init__(self, in_channels, out_channels,
                 m_kernel_size=3, w_kernel_size=3,
                 phi=F.relu, sigma=torch.tanh,
                 sigma_x_compress=10.0, lambda_init=0.0):
        super().__init__()
        self.phi = phi
        self.sigma = sigma
        self.sigma_x_compress = sigma_x_compress
        self.w_kernel_size = w_kernel_size

        # M: the ordinary convolution filter, one per output channel.
        self.m = nn.Parameter(torch.zeros(out_channels, in_channels,
                                          m_kernel_size, m_kernel_size))
        # b: a threshold per output channel. Note it is SUBTRACTED, unlike the
        # bias of nn.Conv2d, which is added.
        self.b = nn.Parameter(torch.zeros(out_channels))
        # lambda: how much the correction counts.
        self.lam = nn.Parameter(torch.full((out_channels,), float(lambda_init)))
        # Omega: the weight of each neighbour. Its shape is
        # (out, in, k, k), so besides pooling over neighbours it also mixes
        # channels. Normalised so that the correction is a weighted average;
        # without that, summing k*k*C saturated differences would swamp U.
        w = torch.ones(out_channels, out_channels, w_kernel_size, w_kernel_size)
        self.w = nn.Parameter(w / w[0].numel(), requires_grad=False)

    def forward(self, x):
        # 1. The ordinary convolution and the threshold. On its own this is a
        #    standard layer.
        u = F.conv2d(x, self.m, padding='same') - self.b.view(1, -1, 1, 1)

        # 2. The differences with the neighbours. unfold pulls out, for each
        #    pixel, the column of values in its window: (B, C, k*k, H*W).
        k = self.w_kernel_size
        neighbours = F.unfold(u, kernel_size=k, padding=k // 2)
        B, C, H, W = u.shape
        neighbours = neighbours.view(B, C, k * k, H * W)
        centre = u.view(B, C, 1, H * W)
        diff = neighbours - centre

        # 3. The inner nonlinearity: it saturates large differences, so the
        #    layer distinguishes small ones well and treats all large ones
        #    alike. This is what cannot be absorbed into a linear filter.
        s = self.sigma(self.sigma_x_compress * diff)

        #    At the borders unfold pads with zeros, and a zero is not a neutral
        #    neighbour here: it would give sigma(0 - centre), which is not zero.
        #    Neighbours that fall outside the image must contribute nothing.
        inside = F.unfold(torch.ones_like(u[:, :1]), kernel_size=k, padding=k // 2)
        s = s * inside.view(B, 1, k * k, H * W)

        # 4. Weighted average of those saturated differences, summing over the
        #    neighbours and over the channels at the same time.
        weights = self.w.view(self.w.size(0), C, k * k)
        correction = torch.einsum('bckp,ock->bop', s, weights).view(B, -1, H, W)

        # 5. Subtract the correction and apply the outer nonlinearity.
        return self.phi(u - self.lam.view(1, -1, 1, 1) * correction)
