# ML Config Management (2026)

*Progressive upgrade path: plain dict → dataclasses → Pydantic v2 → Hydra.*

## Table of Contents

1. [[#1. The Problem with Plain Dicts|1. The Problem with Plain Dicts]]
2. [[#2. Config Structure — Monolithic vs. Nested|2. Config Structure — Monolithic vs. Nested]]
3. [[#3. Tier 1 — dataclasses|3. Tier 1 — dataclasses]]
4. [[#4. Tier 2 — Pydantic v2|4. Tier 2 — Pydantic v2]]
5. [[#5. Tier 3 — Hydra|5. Tier 3 — Hydra]]
6. [[#6. Decision Guide|6. Decision Guide]]
7. [[#7. References|7. References]]

---

## 1. The Problem with Plain Dicts

```python
config = {"lr": 1e-3, "batch_size": 32, "model": {"d_model": 512}}
```

Silent failure modes:

- **Typos are silent:** `config["learing_rate"]` raises `KeyError` at access time, not definition time.
- **No types:** `config["lr"] = "1e-3"` (a string) propagates into your optimizer with no complaint.
- **No validation:** nothing prevents `batch_size = -5` or `n_heads = 7` on a `d_model = 512` transformer.
- **No structure:** nested dicts grow into `config["model"]["attention"]["rope"]["theta"]` — no IDE autocomplete, no documentation.
- **No serialization contract:** dumping to JSON/YAML is ad-hoc; reloading may silently cast types differently.

---

## 2. Config Structure — Monolithic vs. Nested

Before choosing a config tool, choose a config *shape*. The tool is secondary; a poorly structured config is still painful regardless of whether it's a dict, a Pydantic model, or a YAML file.

### 🏗️ Recommendation: nested by concern, flat at root

**Don't** put everything in one flat object:

```python
# Flat/monolithic — becomes unmanageable past ~15 keys
config = TrainConfig(
    d_model=512, n_heads=8, n_layers=6,      # model
    lr=1e-3, batch_size=32, max_steps=100_000, # training
    dataset="imagenet", num_workers=8,          # data
    wandb_project="my-exp", log_every=50,       # logging
)
```

**Do** group by the subsystem that owns each setting:

```python
@dataclass
class ModelConfig:
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6

@dataclass
class TrainingConfig:
    lr: float = 1e-3
    batch_size: int = 32
    max_steps: int = 100_000

@dataclass
class DataConfig:
    dataset: str = "imagenet"
    num_workers: int = 8

@dataclass
class LoggingConfig:
    project: str = "my-exp"
    log_every: int = 50

@dataclass
class Config:                             # ← flat root that composes the rest
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
```

### 📐 Why nested?

**Each component only receives what it needs.** The model constructor takes `ModelConfig`, the dataloader takes `DataConfig`. This is *dependency inversion* applied to configuration: components declare their config requirements explicitly and are testable in isolation.

```python
model = build_model(cfg.model)       # model code never sees lr or dataset
loader = build_dataloader(cfg.data)  # data code never sees d_model
trainer = Trainer(cfg.training)
```

Passing the entire `Config` to every function is the config equivalent of global state — it hides dependencies and makes components hard to reuse.

**Nesting mirrors Hydra's config groups.** Each nested sub-config (`cfg.model`, `cfg.training`) maps directly to a Hydra config group. When you eventually move to Hydra, swapping `model=transformer` for `model=mamba` is a one-line CLI flag rather than a surgery on a single large config file. Nesting now pays dividends later.

**Bounded growth.** Each sub-config stays focused. A `ModelConfig` that grows to 20 keys is a signal to look at your architecture; a monolithic `Config` with 60 keys just looks like noise.

### 🤔 When is monolithic acceptable?

A single flat config is fine when:
- You have ≤ 15 total keys and your project is a self-contained script or notebook.
- No subsystem needs to be reused, tested, or swapped independently.

As soon as you find yourself writing `cfg.d_model` in your data loading code, or filtering keys to pass to a function, it's time to split.

> [!NOTE] How many nesting levels?
> **Two levels is almost always right:** a root `Config` composed of flat sub-configs. Three or more levels (`cfg.model.attention.rope.theta`) add cognitive overhead without much gain. If a sub-config starts needing its own sub-configs, it's often a sign the subsystem's interface is too broad.

### 🔑 Canonical groupings for ML training

| Group | Owns | Example keys |
|-------|------|--------------|
| `model` | Architecture hyperparams | `d_model`, `n_heads`, `n_layers`, `dropout` |
| `training` | Optimization loop | `lr`, `batch_size`, `max_steps`, `grad_clip`, `scheduler` |
| `data` | Dataset and preprocessing | `dataset`, `seq_len`, `num_workers`, `augment` |
| `logging` | Experiment tracking | `project`, `run_name`, `log_every`, `ckpt_dir` |

Add groups when a genuine new owner appears (e.g. `eval`, `distributed`). Don't split `training` just because it has many keys.

---

## 3. Tier 1 — dataclasses

**The trigger:** You're passing a config dict around and can't remember whether the key is `"learning_rate"` or `"lr"`. Your IDE can't help — dict keys are opaque strings. You grep the codebase, find three inconsistent spellings, and discover a silent bug that's been present for weeks. The moment your config has more than a handful of keys and gets passed across module boundaries, you want a *named, typed structure*.

> [!INFO] When to use
> You want structure and dot-access with zero new dependencies. Good starting point for a small project.

```python
from dataclasses import dataclass, field, asdict

@dataclass
class ModelConfig:
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6

@dataclass
class TrainingConfig:
    lr: float = 1e-3
    batch_size: int = 32
    max_steps: int = 100_000
    grad_clip: float = 1.0

@dataclass
class DataConfig:
    dataset: str = "imagenet"
    seq_len: int = 512
    num_workers: int = 8

@dataclass
class LoggingConfig:
    project: str = "my-exp"
    log_every: int = 50
    ckpt_dir: str = "checkpoints"

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

cfg = Config()
print(cfg.model.d_model)       # 512 — dot access
print(cfg.training.lr)         # 0.001
print(asdict(cfg))              # → plain dict for logging

# Each component receives only its sub-config
model = build_model(cfg.model)
loader = build_dataloader(cfg.data)
trainer = Trainer(cfg.training)
```

**What you gain:** type hints (IDE autocomplete, mypy), `__repr__`, `asdict()` for serialization.

**What you don't get:** runtime validation (`cfg.training.lr = "oops"` still silently passes), no YAML/JSON loading, no env var support.

> [!WARNING] Mutable defaults
> Never do `model: ModelConfig = ModelConfig()` as a default value — it creates a shared instance across all `Config` instances. Always use `field(default_factory=ModelConfig)`.

---

## 4. Tier 2 — Pydantic v2

**The trigger:** Someone edits your YAML config so that `lr: 1e-3` becomes `lr: "1e-3"` — now a string. Your `@dataclass` silently accepts it, and AdamW crashes three minutes into training on a remote GPU. Or: you want to reload a saved config to reproduce an experiment, but the round-trip through `asdict()` → JSON → `TrainConfig(...)` requires manual reconstruction. *Dataclasses give you structure but no contract* — no guarantee that values are the right type, in the right range, or even present. The moment you need to trust that a loaded config is valid, you need runtime validation.

> [!INFO] When to use
> Production ML projects where you load config from files or env vars and want runtime validation. The sweet spot for most projects.

Pydantic v2 rewrites the core in Rust, making validation ~5–50x faster than v1.

```python
from pydantic import BaseModel, Field, field_validator, model_validator
import yaml

class ModelConfig(BaseModel):
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6

    @model_validator(mode="after")
    def heads_divide_model(self) -> "ModelConfig":
        assert self.d_model % self.n_heads == 0, \
            f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}"
        return self

class TrainingConfig(BaseModel):
    lr: float = 1e-3
    batch_size: int = 32
    max_steps: int = 100_000
    grad_clip: float = 1.0

    @field_validator("lr")
    @classmethod
    def lr_positive(cls, v: float) -> float:
        assert v > 0, "lr must be positive"
        return v

class DataConfig(BaseModel):
    dataset: str = "imagenet"
    seq_len: int = 512
    num_workers: int = 8

class LoggingConfig(BaseModel):
    project: str = "my-exp"
    log_every: int = 50
    ckpt_dir: str = "checkpoints"

class Config(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
```

### Loading from YAML

```python
with open("config.yaml") as f:
    raw = yaml.safe_load(f)

cfg = Config.model_validate(raw)   # validates + coerces all sub-configs
```

```yaml
# config.yaml
model:
  d_model: 1024
  n_heads: 16
  n_layers: 12
training:
  lr: 0.001
  batch_size: 64
data:
  dataset: imagenet
  num_workers: 16
logging:
  project: my-exp
  ckpt_dir: checkpoints/run-001
```

### Loading from environment variables

```python
from pydantic_settings import BaseSettings

class Config(BaseSettings):
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    model_config = {"env_nested_delimiter": "__"}

# Reads nested env vars: MODEL__D_MODEL=1024, TRAINING__LR=3e-4, DATA__DATASET=wikitext
cfg = Config()
```

### Serialization

```python
cfg.model_dump()            # → nested dict
cfg.model_dump_json()       # → JSON string
cfg.model_dump(mode="json") # → JSON-safe nested dict (no Python-only types)

# Round-trip:
cfg2 = Config.model_validate_json(cfg.model_dump_json())
```

> [!TIP] `model_validate` vs `__init__`
> Always use `model_validate` (or `model_validate_json`) when loading from external sources. Direct `__init__` still runs validators in v2, but `model_validate` also applies coercions (e.g. string `"512"` → int `512`) and is the intended entry point for untrusted input.

---

## 5. Tier 3 — Hydra

**The trigger:** You have a working training script and want to compare Transformer vs. Mamba at three learning rates — six runs total for one ablation table. You edit `config.yaml`, run, edit again, run again. By run four you're no longer sure which checkpoint came from which settings. You rename files by hand, keep a spreadsheet, and still manage to submit the wrong job twice. *Pydantic gives you a validated config object, but it says nothing about how to systematically vary that config across experiments.* The moment you need to define and enumerate a combinatorial experiment space — without editing Python — you need a config composition layer.

> [!INFO] When to use
> Research projects with many experiment variants: you want to swap model architectures, optimizers, or schedulers from the CLI without editing Python. Essential for grid/random hyperparameter sweeps.

Hydra sits on top of **OmegaConf** (its structured config backend). Think of a config as a base document with a sequence of overrides applied:

$$C_{\text{final}} = C_{\text{base}} \oplus \Delta_1 \oplus \Delta_2 \oplus \cdots$$

where each $\Delta_i$ is a YAML group or CLI override. This makes ablations reproducible and auditable.

### Project layout

```
conf/
  config.yaml          ← defaults list
  model/
    transformer.yaml
    mamba.yaml
  training/
    adamw.yaml
    muon.yaml
  data/
    imagenet.yaml
    wikitext.yaml
  logging/
    wandb.yaml
    local.yaml
train.py
```

Each config group corresponds directly to one sub-config from §2's canonical structure. Swapping `model=mamba` replaces the entire `ModelConfig`; `data=wikitext` replaces the entire `DataConfig`.

```yaml
# conf/config.yaml
defaults:
  - model: transformer
  - training: adamw
  - data: imagenet
  - logging: wandb
  - _self_
```

```yaml
# conf/model/transformer.yaml
_target_: myproject.models.Transformer
d_model: 512
n_heads: 8
n_layers: 6
```

```yaml
# conf/training/adamw.yaml
lr: 1e-3
batch_size: 32
max_steps: 100_000
grad_clip: 1.0
```

```yaml
# conf/data/imagenet.yaml
dataset: imagenet
seq_len: 512
num_workers: 8
```

```python
# train.py
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="conf", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))            # full resolved config
    model = hydra.utils.instantiate(cfg.model)   # calls Transformer(**cfg.model)
    loader = build_dataloader(cfg.data)
    trainer = Trainer(cfg.training)
    ...

if __name__ == "__main__":
    train()
```

### CLI overrides

```bash
# Swap model architecture
python train.py model=mamba

# Swap optimizer (entire TrainingConfig)
python train.py training=muon

# Swap dataset
python train.py data=wikitext

# Override individual fields within a group
python train.py training.lr=3e-4 training.batch_size=128

# Multi-run sweep (grid) — 6 runs: 3 lr values × 2 model architectures
python train.py --multirun training.lr=1e-4,3e-4,1e-3 model=transformer,mamba
```

### OmegaConf interpolation

OmegaConf supports `${}` interpolation between config keys, evaluated *lazily on access*:

```yaml
# conf/config.yaml
data:
  root: /data/imagenet
  train: ${data.root}/train      # → /data/imagenet/train
  val: ${data.root}/val          # → /data/imagenet/val

model:
  d_model: 512
  d_ff: ${model.d_model}         # ties d_ff to d_model — change one, both update
```

This is useful for derived quantities (e.g. `d_ff = 4 * d_model`) and avoiding copy-paste errors across config files.

### Structured configs (Pydantic + Hydra)

Hydra's ConfigStore natively uses `@dataclass`, not Pydantic — so there's no single official bridge. Two practical patterns:

**Pattern A — compose then validate (recommended):** Let Hydra compose the config as usual, then validate it with Pydantic at the entry point.

```python
@hydra.main(config_path="conf", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    # Hydra composes; Pydantic validates the full nested Config
    config: Config = Config.model_validate(OmegaConf.to_container(cfg, resolve=True))
    model = build_model(config.model)
    loader = build_dataloader(config.data)
    trainer = Trainer(config.training)
    ...
```

**Pattern B — hydra-zen:** The `hydra-zen` library (MIT Lincoln Lab) auto-generates Hydra-compatible configs from arbitrary Python objects, including Pydantic models.

```python
from hydra_zen import builds, make_config

ModelConf = builds(TransformerModel, d_model=512, n_heads=8, n_layers=6)
TrainingConf = make_config(lr=1e-3, batch_size=32, max_steps=100_000)
DataConf = make_config(dataset="imagenet", num_workers=8)

Config = make_config(model=ModelConf, training=TrainingConf, data=DataConf)

# Register with ConfigStore and use normally with @hydra.main
```

Pattern A is lower-overhead for most projects; hydra-zen pays off when you have deeply nested Python class hierarchies to auto-configure.

> [!WARNING] Hydra's working directory gotcha
> Hydra *used to* change the working directory to a timestamped output folder, silently breaking all relative paths. With `version_base=None` (as in the example above), `hydra.job.chdir` defaults to `False` — this is no longer a problem if you always pass `version_base=None`. If you're on an older setup without it, add `hydra.job.chdir=False` to your config or CLI to opt out explicitly.

### Output directory

Every run writes to `outputs/<date>/<time>/` by default, including:
- `.hydra/config.yaml` — the **fully resolved** config (exact reproducibility)
- `.hydra/overrides.yaml` — the CLI overrides applied
- Your training logs

This is the key reproducibility guarantee: every run is a pure function of its `config.yaml`.

---

## 6. Decision Guide

| Situation | Recommendation |
|-----------|----------------|
| Script or notebook, ≤ 10 config keys | `@dataclass` |
| Production training run, config loaded from file or env | Pydantic v2 |
| Research: ≥ 3 model variants or optimizer variants to compare | Hydra |
| Research: systematic grid/random sweeps | Hydra + `--multirun` |
| You want both validation AND Hydra composition | Compose-then-validate or `hydra-zen` |
| PyTorch Lightning project | `LightningCLI` (uses `jsonargparse` internally) |
| Google JAX ecosystem / TF project | `ml_collections` or `gin-config` |

> [!DANGER] Don't over-engineer early
> Start with `@dataclass`. Migrate to Pydantic v2 once you need validation or YAML loading. Add Hydra only when you have real multi-variant experiments — its abstraction cost is real.

### 🔀 Ecosystem alternatives

| Tool | Ecosystem | Style | When it shines |
|------|-----------|-------|----------------|
| [**jsonargparse**](https://jsonargparse.readthedocs.io/) / LightningCLI | PyTorch Lightning | CLI-first, class-based | Lightning projects; auto-generates CLI from `__init__` signatures |
| [**gin-config**](https://github.com/google/gin-config) | Google / JAX | Decorator-based | Binding hyperparameters to arbitrary functions without config objects |
| [**ml_collections**](https://github.com/google/ml_collections) | Google / JAX | `ConfigDict` | JAX/T5/Flax codebases; supports lazy dict-like access with type safety |
| [**hydra-zen**](https://mit-ll-responsible-ai.github.io/hydra-zen/) | Any | Auto-builds configs | Complex class hierarchies you want Hydra to instantiate |

`gin-config`'s approach is worth understanding: instead of a config object, you bind values to function arguments directly via decorators, so the config *is* the call graph. Very ergonomic for research code that isn't class-structured.

---

## 7. References

| Reference Name | Brief Summary | Link |
|---|---|---|
| Pydantic v2 docs | Full reference for `BaseModel`, validators, settings | [docs.pydantic.dev](https://docs.pydantic.dev/latest/) |
| Hydra docs | Config composition, CLI overrides, multi-run | [hydra.cc/docs](https://hydra.cc/docs/intro/) |
| OmegaConf docs | Structured configs, interpolation, merging | [omegaconf.readthedocs.io](https://omegaconf.readthedocs.io/) |
| pydantic-settings | Env var loading with Pydantic v2 | [docs.pydantic.dev/latest/concepts/pydantic_settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) |
| hydra-zen | Auto-generates Hydra configs from Python objects; Pydantic integration | [mit-ll-responsible-ai.github.io/hydra-zen](https://mit-ll-responsible-ai.github.io/hydra-zen/) |
| jsonargparse | CLI argument parsing that maps directly to class `__init__` signatures | [jsonargparse.readthedocs.io](https://jsonargparse.readthedocs.io/) |
| gin-config | Decorator-based config binding for Google/JAX research code | [github.com/google/gin-config](https://github.com/google/gin-config) |
| ml_collections | Google's ConfigDict for JAX/T5/Flax projects | [github.com/google/ml_collections](https://github.com/google/ml_collections) |
