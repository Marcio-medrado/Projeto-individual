# src/taia_lab/pipelines/supervised_pipeline.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
import numpy as np
import torch
import yaml
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from taia_lab.utils.seed import prefer_device, seed_everything


# -------------------------
# Config (mínimo necessário)
# -------------------------
@dataclass(frozen=True)
class SupervisedPipelineConfig:
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

    # Runtime (opcional)
    deterministic: bool = True
    device_preference: str = "auto"  # auto|cuda|mps|cpu


def project_root() -> Path:
    """Detecta raiz do projeto (pyproject.toml ou .git)."""
    p = Path(__file__).resolve()
    for parent in [p, *p.parents]:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    raise RuntimeError("Não foi possível detectar a raiz do projeto (pyproject.toml/.git).")


def ensure_dirs(root: Path) -> Dict[str, Path]:
    paths = {
        "models": root / "models",
        "artifacts": root / "artifacts",
        "reports": root / "reports",
        "mlruns": root / "mlruns",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _coerce_tags(tags: Any) -> Dict[str, str]:
    if not tags:
        return {}
    return {str(k): str(v) for k, v in dict(tags).items()}


def parse_cfg(cfg: Dict[str, Any]) -> SupervisedPipelineConfig:
    exp = cfg.get("experiment", {}) or {}
    data = cfg.get("data", {}) or {}
    train = cfg.get("train", {}) or {}
    model = cfg.get("model", {}) or {}
    tracking = cfg.get("tracking", {}) or {}
    runtime = cfg.get("runtime", {}) or {}

    name = exp.get("name")
    desc = str(exp.get("description", "")).strip()

    tool = tracking.get("tool", "mlflow")
    if tool != "mlflow":
        raise ValueError("Somente tracking.tool=mlflow é suportado.")
    mlflow_experiment_name = tracking.get("experiment_name")
    tags = _coerce_tags(tracking.get("tags", {}))

    missing = []
    if not name:
        missing.append("experiment.name")
    if not mlflow_experiment_name:
        missing.append("tracking.experiment_name")
    if missing:
        raise ValueError(f"Campos obrigatórios ausentes no YAML: {', '.join(missing)}")

    return SupervisedPipelineConfig(
        name=str(name),
        description=desc,
        seed=int(data.get("seed", 42)),
        n_samples=int(data.get("n_samples", 1200)),
        n_features=int(data.get("n_features", 20)),
        test_size=float(data.get("test_size", 0.2)),
        epochs=int(train.get("epochs", 20)),
        batch_size=int(train.get("batch_size", 64)),
        lr=float(train.get("lr", 0.001)),
        hidden_dim=int(model.get("hidden_dim", 64)),
        n_classes=int(model.get("n_classes", 2)),
        mlflow_experiment_name=str(mlflow_experiment_name),
        tags=tags,
        deterministic=bool(runtime.get("deterministic", True)),
        device_preference=str(runtime.get("device_preference", "auto")).strip().lower(),
    )


# -------------------------
# Modelo
# -------------------------
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


# ============================================================
# Etapas do pipeline (exatamente como no seu diagrama)
# ============================================================
def ingest_data(cfg: SupervisedPipelineConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Ingestão: cria (ou carrega) dados."""
    X, y = make_classification(
        n_samples=cfg.n_samples,
        n_features=cfg.n_features,
        n_informative=max(2, cfg.n_features // 2),
        n_redundant=0,
        n_classes=cfg.n_classes,
        random_state=cfg.seed,
    )
    return X.astype(np.float32), y.astype(np.int64)


def prepare_data(
    cfg: SupervisedPipelineConfig, X: np.ndarray, y: np.ndarray
) -> Tuple[DataLoader, DataLoader, StandardScaler]:
    """Preparação: split + normalização + DataLoader."""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.seed, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, val_loader, scaler


def train_model(
    cfg: SupervisedPipelineConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, float]]:
    """Treinamento: loop treino/val (com logging de métricas por época)."""
    model = TinyMLP(cfg.n_features, cfg.hidden_dim, cfg.n_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    # Caso exista cfg.weight_decay, ele será aplicado aqui (L2 regularization)
    if hasattr(cfg, "weight_decay") and cfg.weight_decay is not None:
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)


    last = {"train_loss": float("nan"), "val_loss": float("nan"), "val_acc": float("nan")}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * xb.size(0)
            n += xb.size(0)

        train_loss = total_loss / max(1, n)
        val_loss, val_acc = evaluate_model(cfg, model, val_loader, device)

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_acc", val_acc, step=epoch)

        last = {"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc}

    return model, last


@torch.no_grad()
def evaluate_model(
    cfg: SupervisedPipelineConfig,
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Avaliação: métricas finais de validação."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in val_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        total_loss += float(loss.item()) * xb.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == yb).sum().item())
        n += xb.size(0)

    return total_loss / max(1, n), correct / max(1, n)


def save_model(
    cfg: SupervisedPipelineConfig,
    model: nn.Module,
    scaler: StandardScaler,
    paths: Dict[str, Path],
    run_id: str,
) -> Dict[str, Path]:
    """Registro local do modelo + pré-processamento (artefatos)."""
    model_path = paths["models"] / f"{cfg.name}_{run_id}.pt"
    scaler_path = paths["artifacts"] / f"{cfg.name}_{run_id}_scaler.json"

    # Modelo (state_dict + config mínima)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_features": cfg.n_features,
            "hidden_dim": cfg.hidden_dim,
            "n_classes": cfg.n_classes,
        },
        model_path,
    )

    # Scaler (JSON simples para ser didático e portátil)
    scaler_payload = {
        "mean_": scaler.mean_.tolist(),
        "scale_": scaler.scale_.tolist(),
        "var_": scaler.var_.tolist(),
        "n_features_in_": int(getattr(scaler, "n_features_in_", cfg.n_features)),
    }
    scaler_path.write_text(json.dumps(scaler_payload, indent=2), encoding="utf-8")

    return {"model_path": model_path, "scaler_path": scaler_path}


def register_artifacts(
    cfg: SupervisedPipelineConfig,
    artifacts: Dict[str, Path],
    paths: Dict[str, Path],
    metrics: Dict[str, float],
    seed_report_notes: str,
) -> Path:
    """Registro no MLflow (artefatos + relatório)."""
    # Artefatos principais
    mlflow.log_artifact(str(artifacts["model_path"]))
    mlflow.log_artifact(str(artifacts["scaler_path"]))

    # Métricas finais como JSON (além das métricas por época)
    metrics_path = paths["artifacts"] / f"{cfg.name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    mlflow.log_artifact(str(metrics_path))

    # Relatório resumido (boa “evidência” para TADS)
    report_path = paths["reports"] / f"report_{cfg.name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
    report_path.write_text(
        "\n".join(
            [
                f"# {cfg.name}",
                "",
                "## Pipeline supervisionado (execução)",
                "- ingestão → preparação → treino → avaliação → registro",
                "",
                "## Métricas finais",
                f"- train_loss: {metrics['train_loss']:.6f}",
                f"- val_loss: {metrics['val_loss']:.6f}",
                f"- val_acc: {metrics['val_acc']:.6f}",
                "",
                "## Artefatos",
                f"- model: {artifacts['model_path']}",
                f"- scaler: {artifacts['scaler_path']}",
                "",
                "## Reprodutibilidade",
                f"- deterministic notes: {seed_report_notes or '(none)'}",
            ]
        ),
        encoding="utf-8",
    )
    mlflow.log_artifact(str(report_path))
    return report_path


# ============================================================
# Orquestração
# ============================================================
def run_pipeline(cfg: SupervisedPipelineConfig) -> None:
    """Executa o fluxo: ingest → prepare → train → eval → save → register."""
    root = project_root()
    paths = ensure_dirs(root)

    # Seeds + device
    seed_report = seed_everything(
        cfg.seed,
        deterministic=cfg.deterministic,
        device_preference=cfg.device_preference,
        set_pythonhashseed=True,
    )
    device_str = prefer_device(cfg.device_preference)
    device = torch.device(device_str)

    # MLflow local
    mlflow.set_tracking_uri(str(paths["mlruns"]))
    mlflow.set_experiment(cfg.mlflow_experiment_name)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{cfg.name}_seed{cfg.seed}_{run_id}"

    with mlflow.start_run(run_name=run_name):
        # Contexto operacional (MLOps)
        mlflow.log_param("runtime.device", device_str)
        mlflow.log_param("runtime.deterministic", cfg.deterministic)
        mlflow.log_param("runtime.device_preference", cfg.device_preference)

        # Parâmetros do pipeline
        mlflow.log_param("experiment.name", cfg.name)
        mlflow.set_tag("description", cfg.description or "")
        for k, v in cfg.tags.items():
            mlflow.set_tag(k, v)

        mlflow.log_param("data.seed", cfg.seed)
        mlflow.log_param("data.n_samples", cfg.n_samples)
        mlflow.log_param("data.n_features", cfg.n_features)
        mlflow.log_param("data.test_size", cfg.test_size)

        mlflow.log_param("train.epochs", cfg.epochs)
        mlflow.log_param("train.batch_size", cfg.batch_size)
        mlflow.log_param("train.lr", cfg.lr)

        mlflow.log_param("model.hidden_dim", cfg.hidden_dim)
        mlflow.log_param("model.n_classes", cfg.n_classes)

        mlflow.set_tag("seed.os", seed_report.os)
        mlflow.set_tag("seed.backend", seed_report.backend)
        if seed_report.notes:
            mlflow.set_tag("seed.notes", seed_report.notes)

        # 1) ingestão
        X, y = ingest_data(cfg)

        # 2) preparação
        train_loader, val_loader, scaler = prepare_data(cfg, X, y)

        # 3) treino (+ logging por época)
        model, last_metrics = train_model(cfg, train_loader, val_loader, device)

        # 4) avaliação final (já retorna no last_metrics, mas mantemos explícito)
        val_loss, val_acc = evaluate_model(cfg, model, val_loader, device)
        last_metrics["val_loss"] = val_loss
        last_metrics["val_acc"] = val_acc

        # 5) salvar modelo
        artifacts = save_model(cfg, model, scaler, paths, run_id)

        # 6) registrar artefatos
        register_artifacts(cfg, artifacts, paths, last_metrics, seed_report.notes)

        print(f"[OK] Pipeline executado: {run_name}")
        print(f"     device={device_str} mlruns={paths['mlruns']}")


def run_supervised_pipeline(cfg: Dict[str, Any] | SupervisedPipelineConfig) -> None:
    """Entrada amigável para a disciplina: recebe dict (YAML parseado) ou config pronta."""
    if isinstance(cfg, SupervisedPipelineConfig):
        run_pipeline(cfg)
    else:
        run_pipeline(parse_cfg(cfg))


# ------------------------------------------------------------
# Opcional: execução direta via arquivo YAML (facilita debug)
# ------------------------------------------------------------
def run_supervised_pipeline_from_yaml(config_path: str) -> None:
    p = Path(config_path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    run_supervised_pipeline(data)