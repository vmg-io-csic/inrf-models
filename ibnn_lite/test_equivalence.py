"""Check that this compact IBNN Lite matches the one in the full library.

Run with:   python3 test_equivalence.py

Needs the full `ibnn` library on the path. If it is not installed the script
says so and exits without failing, so it can be run in any checkout.

The reference is run in exact mode (`calculation_mode='n4'`), which evaluates
sigma for every pair of pixels. Its default mode, `'interpolated'`, replaces
sigma by a sampled approximation to bring the cost down from N^4 to S*N^2, and
is therefore not expected to agree to numerical precision.
"""

import sys

import torch
import torch.nn.functional as F

from ibnn_lite import IBNNLite

try:
    from modified_rf.nn_layers import IBNNLiteLayer
except ImportError:
    sys.exit("The full `ibnn` library is not on the path, so there is nothing "
             "to compare against. Add it with, for example:\n"
             "    PYTHONPATH=/path/to/ibnn/src python3 test_equivalence.py")


CASES = [
    # B, in, out, H,  W,  k, phi
    (1,  1,  1,   8,  8,  3, 'relu'),
    (2,  3,  4,  11, 11,  3, 'relu'),
    (1,  3,  8,  16, 16,  5, 'identity'),
    (3,  2,  2,   7,  9,  3, 'relu'),
    (1,  4,  4,  12, 12,  5, 'identity'),
]


def run_case(B, c_in, c_out, H, W, k, phi_name):
    """Build both layers with the same weights and compare their outputs.

    Inputs
      B, c_in, c_out, H, W, k  ints. Shape of the test case.
      phi_name                 str. 'relu' or 'identity'.

    Outputs
      float. Largest absolute difference between the two outputs.
    """
    torch.manual_seed(B + c_in + c_out + H + k)
    x = torch.randn(B, c_in, H, W)

    reference = IBNNLiteLayer(c_in, c_out, phi_name, k, w_kernel_size=k,
                              in_size=(B, c_in, H, W), initial_lambda=0.7,
                              calculation_mode='n4')
    theta = reference._theta
    with torch.no_grad():
        theta['m'].copy_(torch.randn_like(theta['m']) * 0.3)
        theta['b'].copy_(torch.randn_like(theta['b']) * 0.2)
        theta['w'].copy_(torch.rand_like(theta['w']))
        theta['lambda'].copy_(torch.rand_like(theta['lambda']))

    phi = F.relu if phi_name == 'relu' else (lambda z: z)
    compact = IBNNLite(c_in, c_out, m_kernel_size=k, w_kernel_size=k, phi=phi,
                       sigma_x_compress=float(theta['sigma_x_compress']))
    with torch.no_grad():
        compact.m.copy_(theta['m'])
        compact.b.copy_(theta['b'].view(-1))
        compact.lam.copy_(theta['lambda'].view(-1))
        compact.w.copy_(theta['w'])

    return (reference(x) - compact(x)).abs().max().item()


def main():
    print("  comparing against IBNNLiteLayer in exact mode\n")
    worst = 0.0
    for case in CASES:
        d = run_case(*case)
        worst = max(worst, d)
        B, c_in, c_out, H, W, k, phi = case
        print(f"  B={B} in={c_in} out={c_out} {H}x{W} k={k} phi={phi:9s}"
              f" max difference {d:.2e}")
    print(f"\n  worst case: {worst:.2e}")
    print("  PASS" if worst < 1e-4 else "  FAIL")
    return 0 if worst < 1e-4 else 1


if __name__ == "__main__":
    sys.exit(main())
