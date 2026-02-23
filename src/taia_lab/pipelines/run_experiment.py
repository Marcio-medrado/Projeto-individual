from __future__ import annotations

"""
Runner de experimentos (Aula 03)
- Lê YAML em /configs
- Executa treino (TinyMLP em PyTorch)
- Calcula métricas (val_loss, val_acc)
- Salva artefatos em models/ e reports/
- Registra parâmetros/métricas/artefatos no MLflow (local em ./mlruns)

Uso:
  python -m taia_lab.pipelines.run_experiment --config configs/exp01_baseline.yaml

Visualizar no MLflow:
  mlflow ui --backend-store-uri ./mlruns
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import mlflow
import yaml
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from taia_lab.utils.seed import prefer_device, seed_everything


# -------------------------
# Tipos e Config
# -------------------------
@dataclass(frozen=True)
class ExperimentConfig:
    # Identidade
    name: str
    description: str

    # Dados
    seed: int
    n_samples: int
    n_features: int
    test_size: float

    # Treino
    epochs: int
    batch_size: int
    lr: float

    # Modelo
    hidden_dim: int
    n_classes: int

    # Tracking
    mlflow_experiment_name: str
    tags: Dict[str, str]

    # Execução (opcional; defaults seguros)
    deterministic: bool = True
    device_preference: str = "auto"  # auto | cuda | mps | cpu


# -------------------------
# Utils
# -------------------------
def project_root() -> Path:
    """Detecta raiz do projeto (presença de pyproject.toml ou .git)."""
    p = Path(__file__).resolve()
    for parent in [p, *p.parents]:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    raise RuntimeError("Não foi possível detectar a raiz do projeto (pyproject.toml/.git).")


def ensure_dirs(root: Path) -> Tuple[Path, Path, Path]:
    models_dir = root / "models"
    reports_dir = root / "reports"
    mlruns_dir = root / "mlruns"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    return models_dir, reports_dir, mlruns_dir


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de config não encontrado: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("YAML inválido: esperado um dicionário no topo.")
    return data


def parse_config(y: Dict[str, Any]) -> ExperimentConfig:
    # Campos obrigatórios, com falhas explícitas (didático)
    exp = y.get("experiment", {})
    data = y.get("data", {})
    train = y.get("train", {})
    model = y.get("model", {})
    tracking = y.get("tracking", {})
    runtime = y.get("runtime", {}) or {}  # opcional

    name = exp.get("name")
    description = str(exp.get("description", "")).strip()

    seed = int(data.get("seed"))
    n_samples = int(data.get("n_samples"))
    n_features = int(data.get("n_features"))
    test_size = float(data.get("test_size"))

    epochs = int(train.get("epochs"))
    batch_size = int(train.get("batch_size"))
    lr = float(train.get("lr"))

    hidden_dim = int(model.get("hidden_dim"))
    n_classes = int(model.get("n_classes", 2))

    tool = tracking.get("tool", "mlflow")
    if tool != "mlflow":
        raise ValueError("Somente tracking.tool=mlflow é suportado nesta versão.")
    mlflow_experiment_name = tracking.get("experiment_name")
    tags = tracking.get("tags", {}) or {}
    tags = {str(k): str(v) for k, v in tags.items()}

    # Runtime (opcional): mantém compatibilidade com YAMLs antigos
    deterministic = bool(runtime.get("deterministic", True))
    device_preference = str(runtime.get("device_preference", "auto")).strip().lower()

    missing = []
    if not name:
        missing.append("experiment.name")
    if not mlflow_experiment_name:
        missing.append("tracking.experiment_name")
    if missing:
        raise ValueError(f"Campos obrigatórios ausentes no YAML: {', '.join(missing)}")

    return ExperimentConfig(
        name=str(name),
        description=str(description),
        seed=seed,
        n_samples=n_samples,
        n_features=n_features,
        test_size=test_size,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        hidden_dim=hidden_dim,
        n_classes=n_classes,
        mlflow_experiment_name=str(mlflow_experiment_name),
        tags=tags,
        deterministic=deterministic,
        device_preference=device_preference,
    )


# -------------------------
# Dados e Modelo
# -------------------------
def make_data(cfg: ExperimentConfig):
    X, y = make_classification(
        n_samples=cfg.n_samples,
        n_features=cfg.n_features,
        n_informative=max(2, cfg.n_features // 2),
        n_redundant=0,
        n_classes=cfg.n_classes,
        random_state=cfg.seed,
    )
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.seed, stratify=y
    )
    return (
        torch.tensor(X_train),
        torch.tensor(y_train),
        torch.tensor(X_val),
        torch.tensor(y_val),
    )


class TinyMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_one_epoch(model, loader, loss_fn, opt, device) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()
        total_loss += float(loss.item()) * xb.size(0)
        n += xb.size(0)
    return total_loss / max(1, n)


@torch.no_grad()
def eval_model(model, loader, loss_fn, device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        total_loss += float(loss.item()) * xb.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == yb).sum().item())
        n += xb.size(0)
    avg_loss = total_loss / max(1, n)
    acc = correct / max(1, n)
    return avg_loss, acc


# -------------------------
# Runner
# -------------------------
def run(cfg: ExperimentConfig, config_path: Path) -> None:
    root = project_root()
    models_dir, reports_dir, mlruns_dir = ensure_dirs(root)

    # Determinismo + seeds multi-plataforma (Linux/Windows/macOS + CUDA/MPS)
    seed_report = seed_everything(
        cfg.seed,
        deterministic=cfg.deterministic,
        device_preference=cfg.device_preference,
        set_pythonhashseed=True,
    )

    # Device coerente com preferência e disponibilidade
    device_str = prefer_device(cfg.device_preference)
    device = torch.device(device_str)

    # MLflow local
    mlflow.set_tracking_uri(str(mlruns_dir))
    mlflow.set_experiment(cfg.mlflow_experiment_name)

    X_train, y_train, X_val, y_val = make_data(cfg)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=cfg.batch_size, shuffle=False)

    model = TinyMLP(cfg.n_features, cfg.hidden_dim, cfg.n_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{cfg.name}_{ts}"

    with mlflow.start_run(run_name=run_name):
        # Contexto (relevante para MLOps)
        mlflow.log_param("config_path", str(config_path))
        mlflow.log_param("config_basename", config_path.name)

        # Runtime explícito (o que muda entre Linux/Windows/macOS/MPS/CUDA)
        mlflow.log_param("runtime.device", device_str)
        mlflow.log_param("runtime.deterministic", cfg.deterministic)
        mlflow.log_param("runtime.device_preference", cfg.device_preference)

        # Seed report (bom para auditoria e reprodutibilidade)
        # Como params (consultável em comparação) e notas como tag.
        mlflow.log_param("seed.seed", seed_report.seed)
        mlflow.log_param("seed.os", seed_report.os)
        mlflow.log_param("seed.backend", seed_report.backend)
        mlflow.log_param("seed.deterministic_effective", seed_report.deterministic_effective)
        if seed_report.notes:
            mlflow.set_tag("seed.notes", seed_report.notes)

        # Params do experimento
        for k, v in asdict(cfg).items():
            if k not in {"tags"}:
                mlflow.log_param(k, v)

        # Tags (hipótese, mudança controlada, etc.)
        for k, v in cfg.tags.items():
            mlflow.set_tag(k, v)

        # Descrição (como “justificativa” do experimento)
        if cfg.description:
            mlflow.set_tag("description", cfg.description)

        history = []
        for epoch in range(1, cfg.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, loss_fn, opt, device)
            val_loss, val_acc = eval_model(model, val_loader, loss_fn, device)
            history.append((epoch, train_loss, val_loss, val_acc))

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)

            print(
                f"epoch={epoch}/{cfg.epochs} train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

        # Artefatos (modelo + relatório) — base para “registro de modelos”
        model_path = models_dir / f"{cfg.name}_{ts}.pt"
        torch.save({"state_dict": model.state_dict(), "config": asdict(cfg)}, model_path)

        report_path = reports_dir / f"report_{cfg.name}_{ts}.txt"
        lines = []
        lines.append("TAIA — Aula 03 — Runner de experimentos (YAML + MLflow)\n")
        lines.append(f"device={device}\n")
        lines.append(f"device_str={device_str}\n")
        lines.append(f"deterministic_requested={cfg.deterministic}\n")
        lines.append(f"deterministic_effective={seed_report.deterministic_effective}\n")
        if seed_report.notes:
            lines.append(f"notes={seed_report.notes}\n")
        lines.append(f"config_file={config_path}\n\n")
        for k, v in asdict(cfg).items():
            lines.append(f"{k}={v}\n")
        lines.append("\nEPOCH,TRAIN_LOSS,VAL_LOSS,VAL_ACC\n")
        for (epoch, train_loss, val_loss, val_acc) in history:
            lines.append(f"{epoch},{train_loss:.6f},{val_loss:.6f},{val_acc:.6f}\n")
        lines.append(f"\nmodel_path={model_path}\n")
        report_path.write_text("".join(lines), encoding="utf-8")

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(report_path))

        print(f"\nSaved model:  {model_path}")
        print(f"Saved report: {report_path}")
        print(f"MLflow runs:  {mlruns_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Caminho para YAML do experimento")
    args = parser.parse_args()

    config_path = Path(args.config)
    y = load_yaml(config_path)
    cfg = parse_config(y)
    run(cfg, config_path)


if __name__ == "__main__":
    main()
