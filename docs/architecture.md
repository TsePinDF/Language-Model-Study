# Existing MLX Architecture

This document records the implementation in `main.py` before the PyTorch port.
The source code is treated as authoritative; the README is useful background but
does not define behavior.

## High-Level Flow

The model processes one raw byte at a time. There is no tokenizer.

```text
current byte id
  -> Encoder embedding
  -> RTU layer 0
  -> RTU layer 1
  -> ...
  -> RTU layer N - 1
  -> Decoder byte logits and stop probability
```

At each step the model may also receive the next byte and an `end` flag. When a
next byte is present, the step performs a parameter update immediately. When the
next byte is absent, generation still updates state/traces and may update
weights depending on runtime mode.

The original implementation mixes model parameters, recurrent state, optimizer,
training logic, sampling, checkpointing, and interactive runtime into `Model` and
`Runtime`.

## Model Weights vs Runtime Memory

Slow/global learned state:

- encoder embedding weights, shape `[256, dim]`
- decoder byte projection, shape `[dim, 256]`, plus bias `[256]`
- decoder stop projection, shape `[dim, 1]`, plus bias `[1]`
- per-layer learned raw decay vector, shape `[dim]`
- per-layer layer-norm scale and bias, shape `[dim]`
- per-layer dense weight, shape `[dim, dim]`, no bias
- optimizer state

Fast/runtime memory:

- per-layer recurrent state, shape `[dim]`
- per-layer decay trace, shape `[dim]`
- encoder embedding trace, shape `[256, dim]`

The current MLX code assigns `mx.array` values directly to module attributes for
`decay`, `states`, and `decaytrace`. MLX parameter collection semantics make this
worth verifying on an MLX machine: these arrays may appear in `parameters()` even
though `states` and `decaytrace` are also saved separately and manually assigned.
The PyTorch port separates these concepts explicitly.

## Encoder

`Encoder(dim)` contains:

```text
Embedding(256, dim)
embedtrace = zeros([256, dim])
```

Forward:

```text
x = embed[current_byte]
```

For a scalar byte input, `x` has shape `[dim]`.

## Recurrent Trace Unit

Each `Layer(dim)` contains:

```text
raw_decay   : [dim], initialized to 0
states      : [dim], initialized to 0
decaytrace  : [dim], initialized to 0
LayerNorm(dim)
Linear(dim, dim, bias=False)
SiLU
```

Forward:

```text
decay = sigmoid(raw_decay)
state_t = decay * state_{t-1} + x_t + dummy
y_t = x_t + silu(W * layer_norm(state_t))
```

The layer returns:

```text
y_t, state_t, decay
```

The residual path uses the input `x_t`, not `state_t`.

`dummy` is normally zero. During training it is made differentiable so the code
can obtain `d loss / d state_t` for each layer through `value_and_grad`.

## Decoder

`Decoder(dim)` contains:

```text
Linear(dim, 256)
Linear(dim, 1)
```

Forward:

```text
byte_logits = decode(x)
stop_prob = sigmoid(stop(x))
```

For scalar processing:

```text
byte_logits: [256]
stop_prob  : [1]
```

## Loss Terms

The loss is initialized with a latent variance term:

```text
loss_variance = max(0, 1 - sqrt(var(x) + 1e-4))
```

When `nextb is not None`, three more losses are added:

```text
target_latent = stop_gradient(encoder(next_byte))
loss_latent = mean((x - target_latent)^2)
loss_byte = -logits[next_byte] + logsumexp(logits)
target_stop = 1 if end else 0
loss_stop = mean((stop_prob - target_stop)^2)
```

Total loss in the original code:

```text
loss = loss_variance
if next byte exists:
    loss += loss_latent
    loss += loss_byte
    loss += loss_stop
```

The PyTorch port exposes these as independently weighted components while
keeping default weights at `1.0`.

## Stop-Gradient Behavior

The implementation stops gradients in these places:

- target latent is `stop_gradient(encoder(next_byte))`
- updated encoder embedding trace is stored through `stop_gradient`
- updated per-layer recurrent states are stored through `stop_gradient`
- updated per-layer decay traces are stored through `stop_gradient`

There is no backpropagation through time across byte steps. Instead, the
implementation uses traces and immediate per-byte updates.

## Trace Updates

The model calls `value_and_grad(fwd, argnums=(0, 1))` over trainable parameters
and a list of differentiable zero `dummy` vectors. The gradient with respect to
each dummy is used as an estimate of `d loss / d state_i`.

### Embedding Trace

After the forward/backward calculation:

```text
new_embedtrace = old_embedtrace * decay_0
new_embedtrace[current_byte, :] += 1
```

Before storing the new trace, the embedding gradient is manually augmented:

```text
grad_embedding += dloss_dstate_0 * (old_embedtrace * decay_0)
```

This is elementwise over the latent dimension and broadcast across the 256 byte
rows.

### Decay Trace

For layer `i`:

```text
new_decaytrace_i =
    decay_i * old_decaytrace_i
    + decay_i * (1 - decay_i) * old_state_i
```

The raw decay gradient is replaced, not added:

```text
grad_raw_decay_i = dloss_dstate_i * new_decaytrace_i
```

The layer state is then replaced with `state_t` from the forward pass. Both the
state and trace are detached before storage.

## Optimizer Behavior

`Model` owns an `AdamW` optimizer. A normal call to `Model.__call__` performs:

1. forward/loss closure
2. gradient computation
3. manual trace-gradient edits
4. recurrent state and trace assignment
5. optimizer update
6. MLX evaluation of parameters and optimizer state

This means standard inference, online learning, and state updates are tightly
coupled in the original implementation. `Runtime.chatreadonly` prevents saving
weights to disk, but it still calls the model in the same way unless `notrace`
is selected; therefore weights may change in memory during the session.

`notrace=True` bypasses the training closure, uses zero dummies, does not assign
new persistent states/traces, and does not run the optimizer. It is effectively a
stateless diagnostic mode using the stored state values as read-only inputs.

## Sampling

Sampling computes:

```text
probs = softmax(logits)
entropy = -sum(probs * log(probs + 1e-8)) / log(256)
temp = max(0.1, configured_temp * (1 - configured_temp * entropy))
sample = categorical(logits / temp)
```

The softmax in source code is called as `mx.softmax(output)` without an explicit
axis. The PyTorch port uses the last dimension.

## Runtime Modes

`Runtime` exposes:

- `train`: streams files matching `wikipedia_clean/**/wiki_*`
- `chat`: trains on user input, then repeatedly samples model bytes until stop
- `chatreadonly`: avoids periodic/final disk saves, but does not freeze weights
  in memory
- `chatnotrace`: readonly plus `notrace=True`

The dataset and chat loops use adjacent byte pairs. The `end` flag is true for
the last pair in each encoded line or user input string.

## Checkpoint Contents

`Model.save(path)` writes a safetensors file containing:

- `m.*`: flattened `self.parameters()`
- `o.*`: flattened optimizer state
- `embedtrace`
- `state.{i}` for every layer
- `decaytrace.{i}` for every layer

`Model.load(path)` restores each category when present. Missing checkpoint files
are silently ignored.

The checkpoint does not include explicit configuration, runtime step count,
random number generator state, or dataset position.

## Benchmark Script

`benchmark.py` loads the 4.5M model, freezes it, and trains a separate linear
classification head on CoLA sentence labels. It feeds each byte through the
encoder and layers, manually assigning layer states after each byte. It uses the
final state of the last layer as the sentence representation.

The benchmark has a single shared recurrent state as it iterates examples. It
does not reset layer states between sentences, which may be intentional
continuous processing or an accidental leakage source.

## Parameter Count

Ignoring possible ambiguity around `states` and `decaytrace`, the intended
trainable parameter count is:

```text
embedding       = 256 * dim
per RTU layer   = dim * dim + 3 * dim
decoder         = 256 * dim + 256 + dim + 1

total = layers * dim^2 + (3 * layers + 513) * dim + 257
```

For the original `dim=512`, `layers=16`, this is `4,481,793` parameters.

## Unusual or Ambiguous Behavior

- Runtime memory is stored inside the MLX module instead of passed explicitly.
- `states` and `decaytrace` may be collected as MLX parameters depending on MLX
  module semantics.
- `chatreadonly` prevents saving but likely still mutates in-memory weights.
- `notrace=True` computes transient states but does not store them, making it a
  diagnostic mode rather than normal frozen recurrent inference.
- Decay gradients are overwritten by the trace formula instead of combined with
  autograd's direct decay gradient.
- The first layer's decay vector controls the encoder embedding trace.
- The byte cross-entropy is implemented manually for a single scalar target.
- The stop loss uses MSE over a sigmoid probability rather than BCE over logits.
- The variance loss operates on the final latent and is active even when
  `nextb is None`.
- The original implementation does not batch independent streams.
