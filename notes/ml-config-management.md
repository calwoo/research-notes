# ML Config Management (2026)

*Progressive upgrade path: plain dict → dataclasses → Pydantic v2 → Hydra.*

## Table of Contents

1. [[#1. The Problem with Plain Dicts|1. The Problem with Plain Dicts]]
2. [[#2. Tier 1 — dataclasses|2. Tier 1 — dataclasses]]
3. [[#3. Tier 2 — Pydantic v2|3. Tier 2 — Pydantic v2]]
4. [[#4. Tier 3 — Hydra|4. Tier 3 — Hydra]]
5. [[#5. Decision Guide|5. Decision Guide]]
6. [[#6. References|6. References]]

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

## 2. Tier 1 — dataclasses

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
class TrainConfig:
    lr: float = 1e-3
    batch_size: int = 32
    model: ModelConfig = field(default_factory=ModelConfig)

cfg = TrainConfig()
print(cfg.model.d_model)   # 512 — dot access
print(asdict(cfg))          # → plain dict for logging
```

**What you gain:** type hints (IDE autocomplete, mypy), `__repr__`, `asdict()` for serialization.

**What you don't get:** runtime validation (`cfg.lr = "oops"` still silently passes), no YAML/JSON loading, no env var support.

> [!WARNING] Mutable defaults
> Never do `model: ModelConfig = ModelConfig()` as a default value — it creates a shared instance across all `TrainConfig` instances. Always use `field(default_factory=ModelConfig)`.

---

## 3. Tier 2 — Pydantic v2

> [!INFO] When to use
> Production ML projects where you load config from files or env vars and want runtime validation. The sweet spot for most projects.

Pydantic v2 rewrites the core in Rust, making validation ~5–50x faster than v1.

```python
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings
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

class TrainConfig(BaseModel):
    lr: float = 1e-3
    batch_size: int = 32
    model: ModelConfig = Field(default_factory=ModelConfig)

    @field_validator("lr")
    @classmethod
    def lr_positive(cls, v: float) -> float:
        assert v > 0, "lr must be positive"
        return v
```

### Loading from YAML

```python
with open("config.yaml") as f:
    raw = yaml.safe_load(f)

cfg = TrainConfig.model_validate(raw)   # validates + coerces types
```

```yaml
# config.yaml
lr: 0.001
batch_size: 64
model:
  d_model: 1024
  n_heads: 16
```

### Loading from environment variables

```python
from pydantic_settings import BaseSettings

class TrainConfig(BaseSettings):
    lr: float = 1e-3
    batch_size: int = 32

    model_config = {"env_prefix": "TRAIN_"}

# Reads TRAIN_LR and TRAIN_BATCH_SIZE from environment
cfg = TrainConfig()
```

### Serialization

```python
cfg.model_dump()           # → dict
cfg.model_dump_json()      # → JSON string
cfg.model_dump(mode="json") # → JSON-safe dict (no Python-only types)

# Round-trip:
cfg2 = TrainConfig.model_validate_json(cfg.model_dump_json())
```

> [!TIP] `model_validate` vs `__init__`
> Always use `model_validate` (or `model_validate_json`) when loading from external sources. Direct `__init__` still runs validators in v2, but `model_validate` also applies coercions (e.g. string `"512"` → int `512`) and is the intended entry point for untrusted input.

---

## 4. Tier 3 — Hydra

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
  optimizer/
    adamw.yaml
    muon.yaml
train.py
```

```yaml
# conf/config.yaml
defaults:
  - model: transformer
  - optimizer: adamw
  - _self_

training:
  lr: 1e-3
  batch_size: 32
  max_steps: 100_000
```

```yaml
# conf/model/transformer.yaml
_target_: myproject.models.Transformer
d_model: 512
n_heads: 8
n_layers: 6
```

```python
# train.py
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="conf", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))   # full resolved config
    model = hydra.utils.instantiate(cfg.model)   # calls Transformer(**cfg.model)
    ...

if __name__ == "__main__":
    train()
```

### CLI overrides

```bash
# Swap model architecture
python train.py model=mamba

# Override individual fields
python train.py training.lr=3e-4 training.batch_size=128

# Multi-run sweep (grid)
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
    # Compose with Hydra, validate with Pydantic
    config = TrainConfig.model_validate(OmegaConf.to_container(cfg, resolve=True))
    ...
```

**Pattern B — hydra-zen:** The `hydra-zen` library (MIT Lincoln Lab) auto-generates Hydra-compatible configs from arbitrary Python objects, including Pydantic models.

```python
from hydra_zen import builds, instantiate, make_config

ModelConf = builds(TransformerModel, d_model=512, n_heads=8)
TrainConf = make_config(model=ModelConf, lr=1e-3, batch_size=32)

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

## 5. Decision Guide

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

## 6. References

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
