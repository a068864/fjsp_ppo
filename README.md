# FJSP PPO

Deep Reinforcement Learning for **Flexible Job Shop Scheduling (FJSP)** using:

- Stable-Baselines3 **PPO**
- PyTorch / PyTorch Geometric
- Gymnasium
- CUDA / Apple Metal (MPS), with CPU fallback

The environment state is a PyTorch Geometric `HeteroData` graph. A custom SB3 policy encodes the graph with heterogeneous convolutions, scores machine–operation pairs with an `EdgePredictor` (including compatibility efficiency), and masks invalid actions with `-1e9` before sampling.

**Supported Python:** 3.10–3.13.

Default training/eval is **demo-scale** (`DEBUG_SCALE_ENV`: `5` machines × `3` jobs × `4` avg ops, short PPO run) so training can be verified quickly. For a serious run, pass `--full-scale` (`FULL_SCALE_ENV` `25×15×8` plus default model/PPO sizes). Measure FPS / RSS / GPU first:

```bash
python benchmark.py
python benchmark.py --full-scale --n-env-steps 64 --dummy-vec
```

---

## Installation

### 1. Create an environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install --upgrade pip
```

### 2. Install PyTorch (2.10+)

Pick the build that matches your system ([pytorch.org](https://pytorch.org)):

```bash
# CUDA 12.6 example (Linux / Windows)
pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# macOS (Apple Silicon / Metal MPS) — use the default macOS wheel
pip install torch==2.10.0 torchvision torchaudio

# CPU-only example
pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

On Mac, avoid the `+cpu` index URL if you want Metal acceleration; the default wheel includes MPS.

### 3. Install PyTorch Geometric (2.7+)

```bash
pip install "torch-geometric>=2.7.0,<2.9.0"

# Optional compiled extensions (match your torch + CUDA/CPU tag)
# CUDA 12.6 / Torch 2.10:
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.10.0+cu126.html

# CPU / macOS:
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.10.0+cpu.html
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
# Optional: tests / security audit tooling
pip install -r requirements-dev.txt
```

### 5. Verify

```bash
python -c "import torch; import torch_geometric; import gymnasium; import stable_baselines3; print('ok', torch.__version__, 'cuda', torch.cuda.is_available(), 'mps', getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())"
python -m pytest -q
```

---

## Training

From the project root (`fjsp_ppo/`):

```bash
python train.py
```

Training **starts fresh by default** (`resume=False`). To continue from a trusted local checkpoint:

```bash
python train.py --resume --trust-checkpoint
```

`--trust-checkpoint` is required for any SB3 ZIP load. **SB3/cloudpickle checkpoints are executable input** — only load files you created. This project does **not** claim third-party ZIPs are safe.

Useful overrides:

```bash
# Fewer parallel envs / shorter smoke run
python train.py --n-envs 2 --dummy-vec --total-timesteps 8192

# Documented parallel presets
python train.py --n-envs 8

# Device and seed (auto picks CUDA, then Apple MPS, then CPU)
python train.py --device cuda --seed 123
python train.py --device mps --seed 123

# Larger instance
python train.py --n-machines 10 --n-jobs 8 --avg-ops 6

# Full-scale instance (25×15×8) + default model/PPO sizes
python train.py --full-scale
python evaluate.py --full-scale --trust-checkpoint
python baseline_random.py --full-scale
```

All hyperparameters live in `config.py` (`TrainConfig`, `EnvConfig`, `ModelConfig`, `PPOConfig`).

**Checkpoint compatibility:** Older `latest_model.zip` / `best_model.zip` files are incompatible when env size, model dims, or PPO batching change. Resume also requires a matching `.meta.json` fingerprint. After upgrading, start fresh or delete stale zips under `./checkpoints/`.

### Discrete tick semantics

Each `step()` assigns one operation, then advances simulated time by a fixed `time_step`. Only actual processed work is subtracted from remaining durations / machine workload; unused fractional tick capacity is **not** transferred to the next queued operation in the same tick. Terminal `rollout()` uses the same tick routine.

### Demo defaults (`get_debug_train_config()`)

| Hyperparameter      | Value      |
|---------------------|------------|
| instance            | 5×3×4      |
| n_envs              | 2          |
| hidden_dim          | 64         |
| critic_hidden_dim   | 128        |
| num_layers          | 2          |
| dropout             | 0.0 (required) |
| n_steps             | 256        |
| batch_size          | 64         |
| n_epochs            | 4          |
| total_timesteps     | 32768      |
| resume              | False      |
| tensorboard_log     | `./logs`   |

---

## Evaluation

```bash
python evaluate.py --trust-checkpoint
python evaluate.py --model-path ./checkpoints/best_model.zip --n-episodes 20 --trust-checkpoint
python evaluate.py --stochastic --trust-checkpoint
```

Printed metrics:

- Average reward (± std)
- **Successful-episode makespan** (± std) — only successful episodes
- Episode length (± std)
- Success rate
- Success / failure / timeout counts
- Mean inference time (ms/step)

Path resolution:

- Explicit `--model-path` never falls back if missing.
- The default `./checkpoints/best_model.zip` may fall back to sibling `latest_model.zip`.

---

## Random baseline

```bash
python baseline_random.py
python baseline_random.py --n-episodes 20 --seed 123
python baseline_random.py --n-machines 10 --n-jobs 8 --avg-ops 6
```

Samples uniformly among valid actions from `action_mask`. Empty masks fail fast. Reports the same metric schema as `evaluate.py` (no PPO checkpoint).

## Heuristic baselines

Classic FJSP dispatching rules over the valid action mask (ties → lowest flat action index):

`SPT`, `LPT`, `MWKR`, `LWKR`, `MOR`, `LOR`, `FIFO`, `MFE`, `LFE`, `SQ`, `LWQM`, `ECT`.

```bash
python baseline_heuristic.py --rule SPT
python baseline_heuristic.py --rule MWKR --n-episodes 20 --seed 123
python baseline_heuristic.py --all --n-episodes 5
python baseline_heuristic.py --rule ECT --n-machines 10 --n-jobs 8 --avg-ops 6
```

Requires in-process `GraphDummyVecEnv` (`n_envs=1`). Same metric schema as `evaluate.py` / the random baseline.

## MILP baseline (exact)

Offline PuLP+CBC makespan MILP on each held-out instance (`eval_seed + episode_index`). Uses eligibility, `duration × efficiency` processing times, and the full `precede` DAG. Only **Optimal** solves count as success.

```bash
python baseline_milp.py
python baseline_milp.py --n-episodes 5 --seed 42
python baseline_milp.py --time-limit 30 --n-machines 5 --n-jobs 3 --avg-ops 4
```

Demo-scale instances are the intended target; larger instances may need `--time-limit` and may not prove optimality.

---

## Rollout baseline

Times `policy.predict` + `env.step` and reports FPS, process RSS, CUDA allocation, and GPU util (nvidia-smi when present). Demo-scale stays the train default; use this before `--full-scale`.

```bash
python benchmark.py --n-env-steps 64
python benchmark.py --full-scale --n-env-steps 64 --device cuda
```

---

## TensorBoard

```bash
tensorboard --logdir ./logs
```

Logged signals include policy/value/entropy losses, learning rate, episode reward/length, success rate, successful-episode makespan, gradient norm, and FPS.

---

## Checkpointing

Checkpoints are written under `./checkpoints/`:

| File                    | Meaning                                         |
|-------------------------|-------------------------------------------------|
| `latest_model.zip`      | Most recent training snapshot                   |
| `latest_model.zip.meta.json` | Config fingerprint + save metadata         |
| `best_model.zip`        | Best eval mean reward                           |
| `best_score.json`       | Persisted best score (survives resume)          |

- Saves / evals fire on **completed PPO update** boundaries.
- Final policy is evaluated and saved at training end.
- Resume is **opt-in** (`--resume --trust-checkpoint`) and rejects fingerprint mismatches.

---

## Project structure

```text
fjsp_ppo/
  train.py                 # Training entry point
  evaluate.py              # Evaluation entry point
  baseline_random.py       # Uniform random valid-action baseline
  baseline_heuristic.py    # Classic dispatching-rule baselines
  baseline_milp.py         # Exact makespan MILP (PuLP+CBC)
  benchmark.py             # FPS / RSS / GPU baseline
  config.py                # All hyperparameters + validation
  callbacks.py             # Checkpoint / eval / TB / LR callbacks
  monitor.py               # Episode statistics (append-safe CSV)
  utils.py                 # Seeds, device, unflatten_action, logging
  requirements.txt
  requirements-dev.txt
  README.md
  heuristics/
    dispatch_rules.py      # SPT/LPT/MWKR/... action selection
  solvers/
    milp.py                # FJSP instance extract + MILP solve
  envs/
    fjsp_env.py            # Gymnasium-native FJSP environment
  models/
    edge_predictor.py      # Machine–operation scorer (+ efficiency)
    graph_encoder.py       # HeteroConv encoder
    actor_critic.py        # Graph actor-critic
    sb3_policy.py          # GraphActorCriticPolicy for SB3
    graph_ppo.py           # PPO subclass for HeteroData rollouts
  training/
    make_env.py            # VecEnv factories (Dummy / Subproc)
    evaluate.py            # Shared eval metrics
    eval_cli.py            # Shared evaluate/baseline CLI helpers
    graph_buffer.py        # Object-safe rollout buffer for graphs
    checkpoints.py         # Atomic metadata-backed checkpoint helpers
    benchmark.py           # Rollout FPS / RSS / GPU snapshot
  logs/
  checkpoints/
  tests/
```

### Design notes

- **Native Gymnasium env:** `envs/fjsp_env.py` implements `FJSPEnv` directly.
- **No graph flattening:** observations carry slim `HeteroData` plus an action mask.
- **SB3 integration:** `GraphPPO` + `GraphDictRolloutBuffer` store graphs as objects.
- **Action encoding:** `action = machine_id * n_operations + operation_id`.
- **Invalid actions:** logits masked with `-1e9`; empty masks raise (value-only bootstrap still allowed).
- **Deterministic PPO:** policy dropout must be `0.0`.

---

## Troubleshooting

### Windows + `SubprocVecEnv`

Always launch via `python train.py` (the `if __name__ == "__main__"` guard is required for spawn). If subprocess workers hang:

```bash
python train.py --dummy-vec --n-envs 2
```

### CUDA OOM / slow starts

- Keep env tensors on CPU (default in `make_env`); only the policy uses GPU.
- Lower `--n-envs`, `hidden_dim`, `n_steps`, or `batch_size`.

### Resume / trust errors

- Pass `--resume --trust-checkpoint` only for local checkpoints you trust.
- Fingerprint mismatches mean config drifted; start fresh or restore the matching config.

### Empty valid-action mask

Empty masks terminate episodes / fail evaluation. Check gridlock and dependency readiness.

---

## License / attribution

Scheduling dynamics live in `envs/fjsp_env.py` as a Gymnasium-native FJSP environment for Stable-Baselines3 PPO training.
