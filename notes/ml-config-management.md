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
from pydantic import BaseModel, field_validator, model_validator
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
    model: ModelConfig = ModelConfig()

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

### Structured configs (Pydantic + Hydra)

You can combine Pydantic v2 validation with Hydra composition using `hydra-pydantic`:

```python
from hydra_pydantic import PydanticDataClass
from pydantic.dataclasses import dataclass as pydantic_dataclass
from omegaconf import MISSING

@pydantic_dataclass
class ModelConfig:
    d_model: int = MISSING
    n_heads: int = 8
```

> [!WARNING] Hydra's main gotcha
> Hydra changes the working directory to a timestamped output folder by default (`outputs/2026-05-03/14-32-11/`). All relative file paths in your training code break silently. Either use `hydra.utils.get_original_cwd()` or disable with `hydra.job.chdir=False` in your config.

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
| You want both validation AND Hydra composition | Pydantic dataclasses + `hydra-pydantic` |

> [!DANGER] Don't over-engineer early
> Start with `@dataclass`. Migrate to Pydantic v2 once you need validation or YAML loading. Add Hydra only when you have real multi-variant experiments — its abstraction cost is real.

---

## 6. References

| Reference Name | Brief Summary | Link |
|---|---|---|
| Pydantic v2 docs | Full reference for `BaseModel`, validators, settings | [docs.pydantic.dev](https://docs.pydantic.dev/latest/) |
| Hydra docs | Config composition, CLI overrides, multi-run | [hydra.cc/docs](https://hydra.cc/docs/intro/) |
| OmegaConf docs | Structured configs, interpolation, merging | [omegaconf.readthedocs.io](https://omegaconf.readthedocs.io/) |
| pydantic-settings | Env var loading with Pydantic v2 | [docs.pydantic.dev/latest/concepts/pydantic_settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) |
| hydra-pydantic | Integration layer for Pydantic v2 + Hydra structured configs | [github.com/tky823/hydra-pydantic](https://github.com/tky823/hydra-pydantic) |
