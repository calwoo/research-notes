# ML Local Dev Environment Design
*2026-04-29*

## Context

Personal ML dev environment for a 30-week deep learning engineering curriculum (LM-focused). Hardware profile: Apple Silicon Mac for notebooks and code writing; Lambda Labs / RunPod / Vast.ai cloud NVIDIA GPUs for training runs. Editor: VS Code. Starting fresh — no existing setup to migrate.

## Approach

**uv-first, no conda.** A single tool (`uv`) handles Python version management, virtual environments, and package installation across both Mac and cloud. CUDA management is a non-issue: cloud VMs ship with CUDA pre-installed; PyTorch bundles its own CUDA runtime for everything else. The one complexity — Mac needs a CPU/MPS PyTorch wheel, cloud needs a CUDA wheel — is handled declaratively via `pyproject.toml` platform markers.

---

## 1. Package Management & Environment Structure

### Installation

Install uv once, globally:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install and pin a Python version:

```bash
uv python install 3.12
```

### Per-project layout

Every ML project is a `pyproject.toml` project:

```
my-project/
├── pyproject.toml    # dependency declarations
├── uv.lock           # exact lockfile — committed to git
├── .venv/            # auto-created by uv — gitignored
└── src/ or scripts/
```

### Core workflow

```bash
uv add torch              # add dep → updates pyproject.toml + uv.lock
uv sync                   # install from lockfile (creates .venv if absent)
uv run python train.py    # run in project env without activating
```

### Mac vs. cloud PyTorch split

Mac needs the CPU/MPS wheel; cloud (Linux) needs the CUDA wheel. Declare both in `pyproject.toml`:

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
url = "https://download.pytorch.org/whl/cu124"  # adjust to match VM's CUDA version (check with `nvcc --version`)
explicit = true
```

`uv sync` on both machines, different wheels resolved per platform, one committed lockfile.

---

## 2. Notebooks

### Primary: Marimo

```bash
uv add --dev marimo
uv run marimo edit notebook.py
```

Marimo notebooks are pure `.py` files — no JSON, clean git diffs, executable as scripts. Reactive execution model eliminates stale-state bugs. For throwaway experiments:

```bash
uv run marimo edit --sandbox notebook.py
```

`--sandbox` creates an isolated uv env scoped to that file, auto-installs inline deps. Use for one-off explorations that shouldn't pollute project deps.

### Secondary: VS Code Jupyter (`.ipynb`)

For notebooks that need to be shared in Jupyter format:

```bash
uv add --dev ipykernel
```

VS Code's Jupyter extension auto-discovers `.venv` as a kernel — no `ipykernel install --user` needed. Open `.ipynb` → kernel picker → select `.venv`.

### Practical split

| Use case | Tool |
|---|---|
| EDA, debugging training runs, visualizing metrics | Marimo |
| Sharing with collaborators expecting `.ipynb` | VS Code Jupyter |

---

## 3. Cloud GPU Workflow

### Mode A — VS Code Remote SSH (interactive work)

Add the cloud VM to `~/.ssh/config`:

```
Host runpod-gpu
  HostName <pod-ip>
  User root
  IdentityFile ~/.ssh/id_ed25519
  ServerAliveInterval 60
```

Connect via VS Code → Remote Explorer. Local editor, local extensions, remote kernel and filesystem. Jupyter notebooks execute on the GPU directly.

First-time setup on the cloud VM:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <repo>
cd <repo>
uv sync   # installs CUDA PyTorch via linux platform marker
```

### Mode B — git push + run (fire-and-forget training)

```bash
# local
git push

# cloud VM (in tmux)
git pull && uv sync && uv run python train.py
```

Use `tmux` or `screen` so the job survives SSH disconnection. Pipe logs to wandb or a file.

### Provider notes

| Provider | Recommended mode | Notes |
|---|---|---|
| Lambda Labs | Mode A | Persistent storage, stable IPs |
| RunPod / Vast.ai | Mode B | Treat VM as ephemeral, keep everything in git |

---

## 4. VS Code Setup

### Extensions

| Extension ID | Purpose |
|---|---|
| `ms-python.python` | Interpreter management, auto-discovers `.venv` |
| `ms-python.pylance` | Type checking, autocomplete |
| `ms-toolsai.jupyter` | `.ipynb` support |
| `charliermarsh.ruff` | Linting + formatting (replaces flake8 + black) |
| `ms-vscode-remote.remote-ssh` | Cloud VM development |

### Workspace settings (`.vscode/settings.json`)

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "editor.formatOnSave": true,
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff"
  }
}
```

VS Code picks up `.venv` as both the Python interpreter and the Jupyter kernel automatically.

### Ruff config (in `pyproject.toml`)

```toml
[tool.ruff]
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]   # pycodestyle, pyflakes, isort, pyupgrade
```

---

## Summary

| Layer | Tool | Replaces |
|---|---|---|
| Python version management | `uv python install` | pyenv, conda |
| Virtual environments | `uv venv` / `uv sync` | venv, conda envs |
| Package installation | `uv add` | pip, conda install |
| Dependency locking | `uv.lock` | pip-tools, conda lock |
| Notebooks (primary) | Marimo | JupyterLab |
| Notebooks (secondary) | VS Code Jupyter | JupyterLab |
| Linting + formatting | Ruff | flake8 + black + isort |
| Remote dev | VS Code Remote SSH | SSH + manual sync |
