# Two bio-inspired INRF models: IBNN Lite and INME

Minimal, runnable extracts of two models developed at the Vision Modeling Group
(CSIC), put side by side so that the code can be read and run without wading
through either full repository.

Both belong to the same family. A unit's response is **not** a filtered version
of its input: it is a filtered version **minus a nonlinear function of how the
input differs from itself across the neighbourhood**. Writing the two models in
the same notation makes the shared core obvious:

```
IBNN Lite    Y(p) = Phi( U(p) - lambda * SUM_q  Omega(q)  sigma( U(q) - U(p) ) )

INME         B(p,t) = (m*A)(p,t) - lambda * SUM_u T(u) SUM_q w(q) f( A(q,t-u) - (m*A)(p,t) )
```

Same shape, different ingredients. That is the point of this repository: the
two were built for different problems and the differences are substantive, but
the operation underneath is one operation.

## How they differ

| | IBNN Lite | INME |
|---|---|---|
| Domain | **space only**, image to image | **space and time**, movie to a response over space and time |
| Spatial filter on the lateral branch | uniform average by default (Gaussian available) | **Gabor**, tuned to cardinal directions |
| Inner nonlinearity | **sigmoid** (tanh): odd, monotonic | **even and non-monotonic**, positive, zero at the origin |
| Temporal filtering | none | band-pass at the input, low-pass inside the sensor, low-pass at the readout |
| Output | an image, same size as the input | one vector per stimulus, after a spatial and temporal readout |
| Parameters | **learned** by gradient descent | **set by hand**, per experiment |
| Implementation | PyTorch, GPU | MATLAB, CPU |

The last two rows matter as much as the first: IBNN Lite is a trainable layer
you drop into a network, INME is a model whose parameters are chosen to match
physiology and psychophysics. Unifying them means more than merging files.

## IBNN Lite

`ibnn_lite/ibnn_lite.py` is a compact reimplementation, about 50 lines of
actual computation, written to be read in one sitting. It is numerically
equivalent to `IBNNLiteLayer` of the full library, verified to 1.5e-05 across
batch sizes, channel counts, image sizes and kernel sizes.

```bash
cd ibnn_lite
python3 demo_ibnn_lite.py
```

The demo feeds the layer a luminance step. With `lambda = 0` the layer is
exactly a convolution and the step comes out unchanged. With `lambda = 1` the
correction switches on:

```
input                 0.00  0.00  0.00  0.00 | 0.00  1.00  1.00  1.00
output, lambda = 1    0.00  0.00  0.00  0.00 |-0.33  1.33  1.00  1.00
```

Uniform regions are untouched and the two pixels flanking the step are pushed
apart, the bright side enhanced and the dark side suppressed. That asymmetry
around a luminance step is what produces Mach bands in human vision, and it
falls out of the architecture rather than being put in by hand.

To check the equivalence against the full library yourself:

```bash
PYTHONPATH=/path/to/ibnn/src python3 test_equivalence.py
```

## INME

`inme/` holds the five files of the engine, which are self-contained, plus a
stimulus generator and two demos.

### The quick one

```bash
cd inme
matlab -batch "demo_inme_1d"
```

Runs in about four seconds. A grating drifts rightward and then leftward along
one spatial axis, and the model returns a signed response:

```
  direction    response
  ------------------------------
  rightward    +273.2385
  leftward     -273.2325
  ------------------------------
  sum of the two: +0.0060  (should be near zero)
```

Equal and opposite, as they should be: the two stimuli differ only in the sign
of the velocity. The sign of the response *is* the direction the model reports.

### The two-dimensional one

```bash
cd inme
matlab -batch "demo_inme"
```

Runs in about fifteen seconds and recovers all three directions exactly:

```
  stimulus     predicted    error
  ------------------------------------
       0.0 deg     -0.0 deg    0.00 deg
      45.0 deg     45.0 deg    0.00 deg
      90.0 deg     90.0 deg    0.00 deg
```

The structure of the computation is the thing to look at: the input is a movie, the model
returns two motion-opponent channels that vary over space and time, and a
readout collapses them into one vector per stimulus. That readout stage is
what IBNN Lite, having no temporal dimension, does not have.

Scale is worth noting, because it is one of the reasons to want a unified
implementation. These demos are small on purpose. At the size used in the
paper, 300 x 300 px and 400 frames, the same sweep takes over ten minutes: the
2D engine evaluates the nonlinear sensor at every pixel of every frame, in
MATLAB, on CPU. A GPU implementation is the obvious next step.

| file | what it is |
|---|---|
| `stINRF3_compute.m` | the model, 2D |
| `stINRF3_compute_1D.m` | the model, for stimuli that vary along one spatial axis |
| `stINRF3_config.m` | the parameters of every experiment, in one place |
| `stINRF3_readout_gaussian.m`, `..._1D.m` | the spatial readout |
| `make_grating.m` | a drifting sinusoidal grating |

## What is not here

This is an extract, not a replacement. Left out: the eleven experiments and
their stimuli, the evaluation of five optical-flow networks, the figures, the
fast approximate computation mode, the other layers of the INRF family and
everything to do with training classifiers.

| | full repository |
|---|---|
| IBNN | `github.com/vmg-io-csic/ibnn` |
| INME | `github.com/vmg-io-csic/INME` |

## License

Apache License 2.0, the same as the `ibnn` repository. See `LICENSE`.
