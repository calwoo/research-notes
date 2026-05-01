# Local ML Dev Environment (2026)

*Mac + cloud GPU setup using uv. No conda.*

## Table of Contents

1. [[#1. Package Management|1. Package Management]]
2. [[#2. Per-Project Layout|2. Per-Project Layout]]
3. [[#3. Mac vs. Cloud PyTorch|3. Mac vs. Cloud PyTorch]]
4. [[#4. Notebooks|4. Notebooks]]
5. [[#5. VS Code Setup|5. VS Code Setup]]
6. [[#6. Cloud GPU Workflow|6. Cloud GPU Workflow]]

---

## 1. Package Management

Install `uv` once, globally — it replaces pip, venv, and pyenv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
```

Core commands:

```bash
uv add torch          # add dep → updates pyproject.toml + uv.lock
uv sync               # install from lockfile, create .venv if absent
uv run python train.py  # run in env without activating
```

`uv add` requires a `pyproject.toml` in the current directory or a parent — if none exists, run `uv init` first. For a one-off script without a project:

```bash
uv run --with torch python script.py   # temporary isolated env, nothing persists
```

For CLI tools that should be available globally on PATH (think pipx):

```bash
uv tool install ruff    # ruff binary available globally
uv tool install httpie
```

`uv tool install` is for executables only — libraries like torch have no global install path in uv and are always project-scoped.

> [!INFO] Why not conda?
> The main reason to use conda for ML was managing CUDA/cuDNN. In 2026 this is moot: cloud VMs (Lambda, RunPod) ship with CUDA pre-installed, and PyTorch bundles its own CUDA runtime. The only exception is if you need packages only available on conda channels — rare for standard ML work.

---

## 2. Per-Project Layout

```
my-project/
├── pyproject.toml    ← dependency declarations
├── uv.lock           ← exact lockfile, committed to git
├── .venv/            ← auto-created by uv, gitignored
└── src/
```

The `uv.lock` file encodes exact versions for *all platforms simultaneously* — one lockfile works for both Mac and Linux cloud VMs.

---

## 3. Mac vs. Cloud PyTorch

Mac needs the CPU/MPS wheel; Linux cloud needs the CUDA wheel. Declare both in `pyproject.toml` via platform markers:

```toml
[project]
dependencies = ["torch", "torchvision", "torchaudio"]

[tool.uv.sources]
torch = [
  { index = "pytorch-cpu", marker = "sys_platform == 'darwin'" },
  { index = "pytorch-cu124", marker = "sys_platform == 'linux'" },
]
torchvision = [
  { index = "pytorch-cpu", marker = "sys_platform == 'darwin'" },
  { index = "pytorch-cu124", marker = "sys_platform == 'linux'" },
]
torchaudio = [
  { index = "pytorch-cpu", marker = "sys_platform == 'darwin'" },
  { index = "pytorch-cu124", marker = "sys_platform == 'linux'" },
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true
```

> [!WARNING] Check your CUDA version
> `cu124` targets CUDA 12.4. Verify with `nvcc --version` on the cloud VM and adjust the index URL accordingly (e.g. `cu121`, `cu126`).

Same `uv sync` command on both machines — different wheels, same lockfile.

---

## 4. Notebooks

### Marimo (primary)

*Reactive notebooks as pure `.py` files — no JSON, clean git diffs, runnable as scripts.*

```bash
uv add --dev marimo
uv run marimo edit notebook.py
```

For throwaway experiments isolated from project deps:

```bash
uv run marimo edit --sandbox notebook.py
```

`--sandbox` creates a scoped uv env for that file and auto-installs inline dep declarations. Use this for one-off EDA that shouldn't pollute `pyproject.toml`.

> [!TIP] The reactive model
> Marimo cells re-execute automatically when their inputs change. This eliminates the core notebook pathology: stale state from running cells out of order. The tradeoff is that long-running cells (model training) need to be wrapped in explicit guards.

### VS Code Jupyter (secondary)

For `.ipynb` files you need to share:

```bash
uv add --dev ipykernel
```

VS Code auto-discovers `.venv` as a kernel — no `ipykernel install --user` needed. Open `.ipynb` → kernel picker → select `.venv`.

| Use case | Tool |
|---|---|
| EDA, debugging runs, metric visualization | Marimo |
| Sharing with collaborators expecting `.ipynb` | VS Code Jupyter |

---

## 5. VS Code Setup

**Extensions:**

| Extension | Purpose |
|---|---|
| `ms-python.python` | Interpreter management, auto-discovers `.venv` |
| `ms-python.pylance` | Type checking, autocomplete |
| `ms-toolsai.jupyter` | `.ipynb` support |
| `charliermarsh.ruff` | Linting + formatting (replaces flake8 + black + isort) |
| `ms-vscode-remote.remote-ssh` | Cloud VM development |

**Workspace `.vscode/settings.json`:**

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "editor.formatOnSave": true,
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff"
  }
}
```

**Ruff config in `pyproject.toml`:**

```toml
[tool.ruff]
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]
```

---

## 6. Cloud GPU Workflow

### Mode A — VS Code Remote SSH

Best for interactive work (debugging, notebook sessions). Add the VM to `~/.ssh/config`:

```
Host runpod-gpu
  HostName <pod-ip>
  User root
  IdentityFile ~/.ssh/id_ed25519
  ServerAliveInterval 60
```

Connect via VS Code → Remote Explorer. Your editor runs locally; the kernel and filesystem are on the GPU box.

First-time VM setup:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <repo> && cd <repo>
uv sync    # installs CUDA PyTorch via linux platform marker
```

### Mode B — git push + tmux

Best for fire-and-forget training runs. Treat the VM as ephemeral — keep everything in git.

```bash
# local
git push

# cloud VM
git pull && uv sync && tmux new-session -d -s train 'uv run python train.py'
```

Use `tmux` so the job survives SSH disconnection.

| Provider | Mode | Notes |
|---|---|---|
| Lambda Labs | A or B | Persistent storage, stable IPs |
| RunPod / Vast.ai | B | Treat VM as ephemeral |

> [!INFO] Why uv.lock makes this reliable
> The lockfile resolves exact versions for both `darwin` and `linux` at `uv lock` time. `uv sync` on the cloud VM installs the exact CUDA wheel that was resolved on your Mac — not whatever is latest on the index that day.
