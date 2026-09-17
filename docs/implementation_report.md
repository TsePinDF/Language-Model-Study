# Implementation Report

## Files Added or Modified

Modified:

- `.gitignore`
- `README.md`

Added:

- `docs/architecture.md`
- `docs/implementation_report.md`
- `tmt/` PyTorch package
- `configs/tmt_5m.yaml`
- `configs/tmt_25m.yaml`
- `configs/tmt_50m.yaml`
- `configs/tmt_100m.yaml`
- `scripts/train.py`
- `scripts/generate.py`
- `scripts/evaluate.py`
- `scripts/download_dataset.py`
- `scripts/convert_checkpoint.py`
- `streamlit_app.py`
- `tests/`

The original MLX files `main.py` and `benchmark.py` were preserved unchanged.

## Architecture Interpretation

The MLX source implements a one-byte-at-a-time recurrent model:

```text
byte -> embedding -> RTU stack -> byte logits + stop probability
```

Each RTU keeps a persistent state and learned raw decay vector:

```text
decay = sigmoid(raw_decay)
state_t = decay * state_{t-1} + x_t + dummy
x_{t+1} = x_t + silu(W layer_norm(state_t))
```

The training objective combines latent variance, latent next-byte embedding MSE,
manual byte cross entropy, and stop-probability MSE. The implementation uses
dummy state perturbations to obtain per-layer `d loss / d state` values and then
manually applies embedding-trace and decay-trace gradient corrections.

The PyTorch port keeps trainable weights in `TMTModel` and runtime memory in
explicit `TMTState` objects.

## Differences From MLX

- Runtime state and traces are explicit instead of stored invisibly inside the
  module.
- Training/update behavior lives in `TMTTrainer`, not `model.forward()`.
- Batch dimensions are supported for independent streams.
- Loss components are exposed and weighted through configuration.
- Checkpoints distinguish model-only state from full training state.
- All new checkpoints use pickle-free safetensors storage.
- Training checkpoints are grouped by run ID and include stream cursors, reset
  counters, optimizer state, runtime memory, and RNG state for exact resume.
- CUDA/BF16 support is implemented through PyTorch autocast and verified on RTX
  A6000 hardware.
- A Streamlit console launches and monitors persisted background jobs.

## Unresolved Ambiguities

- MLX may collect direct `mx.array` attributes such as `states` and
  `decaytrace` as parameters; this should be verified on an MLX machine.
- Exact MLX/PyTorch numerical parity has not been measured.
- The original `chatreadonly` mode appears to avoid saving but may still mutate
  in-memory weights.
- The original CoLA benchmark does not reset recurrent state between examples.
- Trace-gradient behavior was ported as written, including replacement of decay
  gradients rather than addition to direct autograd decay gradients.

## Tests Performed

Verified locally:

```text
python -m unittest discover -v
```

Result:

```text
20 tests run
20 passed
```

Additional smoke checks:

```text
python scripts/train.py --config configs/tmt_5m.yaml --data README.md --max-steps 1 --device cuda:0
python scripts/generate.py --config configs/tmt_5m.yaml --prompt Hi --max-new-bytes 2 --device cuda:0
```

The training smoke printed matching actual and expected parameter counts for the
5M config.

## Parameter Counts

| Config | Actual trainable parameters |
| --- | ---: |
| `tmt_5m` | 4,481,793 |
| `tmt_25m` | 24,713,473 |
| `tmt_50m` | 49,924,097 |
| `tmt_100m` | 100,635,393 |

## CUDA

CUDA 12.8 PyTorch execution and BF16 autocast were verified with four visible
RTX A6000 GPUs. The current scripts select one device per training process; they
do not yet implement distributed data parallel training.

## Recommended Next Experiment

Run the 5M PyTorch config on a CUDA machine and compare against the MLX
implementation at the level of tensor shapes, loss components, trace updates,
and short qualitative rollouts. After that, train `tmt_25m` with frozen-inference
and online-learning evaluation checkpoints to see whether retention and
plasticity improve before moving to `tmt_50m` or `tmt_100m`.

## Technical Risks Before Scaling Beyond 100M

- Trace-gradient math may not be an exact RTRL formulation.
- Runtime embedding traces scale with `batch_size * 256 * dim`.
- Per-byte Python training loops may become a throughput bottleneck.
- Online weight updates during generation can make evaluation hard to reproduce.
- Reset policies and document boundaries need dataset-specific validation.
- Effective memory horizon must be measured; no fixed context window does not
  imply infinite usable memory.
