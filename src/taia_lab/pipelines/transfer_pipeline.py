from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import mlflow
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

import matplotlib.pyplot as plt

from taia_lab.utils.seed import prefer_device, seed_everything


# -------------------------
# Config
# -------------------------
@dataclass(frozen=True)
class TransferPipelineConfig:
    # Identidade
    name: str
    description: str

    # Dados
    dataset: str
    data_dir: str
    seed: int
    val_split: float
    num_workers: int
    image_size: int

    # Treino
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float

    # Modelo
    strategy: str  # baseline|feature_extraction|finetune
    backbone: str  # mobilenet_v3_small
    pretrained: bool
    finetune_layers: int
    n_classes: int

    # Tracking
    mlflow_experiment_name: str
    tags: Dict[str, str]

    # Runtime
    deterministic: bool = True
    device_preference: str = "auto"  # auto|cuda|mps|cpu


def project_root() -> Path:
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


def parse_cfg(cfg: Dict[str, Any]) -> TransferPipelineConfig:
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

    dataset_name = str(data.get("dataset", "cifar10")).strip().lower()
    if dataset_name != "cifar10":
        raise ValueError("Neste laboratório, somente data.dataset=cifar10 é suportado.")

    strategy = str(model.get("strategy", "feature_extraction")).strip().lower()
    if strategy not in {"baseline", "feature_extraction", "finetune"}:
        raise ValueError("model.strategy deve ser baseline|feature_extraction|finetune.")

    backbone = str(model.get("backbone", "mobilenet_v3_small")).strip().lower()
    if backbone != "mobilenet_v3_small":
        raise ValueError("Neste laboratório, somente model.backbone=mobilenet_v3_small é suportado.")

    return TransferPipelineConfig(
        name=str(name),
        description=desc,
        dataset=dataset_name,
        data_dir=str(data.get("data_dir", "data")),
        seed=int(data.get("seed", 42)),
        val_split=float(data.get("val_split", 0.2)),
        num_workers=int(data.get("num_workers", 2)),
        image_size=int(data.get("image_size", 224)),
        epochs=int(train.get("epochs", 8)),
        batch_size=int(train.get("batch_size", 64)),
        lr=float(train.get("lr", 0.001)),
        weight_decay=float(train.get("weight_decay", 1e-4)),
        strategy=strategy,
        backbone=backbone,
        pretrained=bool(model.get("pretrained", True)),
        finetune_layers=int(model.get("finetune_layers", 0)),
        n_classes=int(model.get("n_classes", 10)),
        mlflow_experiment_name=str(mlflow_experiment_name),
        tags=tags,
        deterministic=bool(runtime.get("deterministic", True)),
        device_preference=str(runtime.get("device_preference", "auto")).strip().lower(),
    )


# -------------------------
# Utilidades de logging
# -------------------------
def count_trainable_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def log_loss_curves_as_artifact(
    cfg: TransferPipelineConfig,
    paths: Dict[str, Path],
    train_losses: list[float],
    val_losses: list[float],
    run_id: str,
) -> Path:
    fig = plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="train_loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"Loss Curves - {cfg.name}")
    plt.legend()

    out_path = paths["reports"] / f"loss_curves_{cfg.name}_{run_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    mlflow.log_artifact(str(out_path))
    return out_path


# ============================================================
# Etapas do pipeline
# ============================================================
def ingest_data(cfg: TransferPipelineConfig) -> Tuple[Dataset, Dataset]:
    """Ingestão: obtém o dataset bruto (download automático se necessário)."""
    # Normalização de ImageNet para compatibilidade com pesos pré-treinados
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    tfm = transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    root = Path(cfg.data_dir)
    train_ds = datasets.CIFAR10(root=str(root), train=True, download=True, transform=tfm)
    test_ds = datasets.CIFAR10(root=str(root), train=False, download=True, transform=tfm)
    return train_ds, test_ds


def prepare_data(
    cfg: TransferPipelineConfig, train_ds: Dataset
) -> Tuple[DataLoader, DataLoader]:
    """Preparação: split train/val + DataLoaders."""
    n = len(train_ds)
    n_val = int(round(cfg.val_split * n))
    n_train = n - n_val
    if n_val <= 0 or n_train <= 0:
        raise ValueError("val_split inválido: precisa deixar exemplos para treino e validação.")

    g = torch.Generator().manual_seed(cfg.seed)
    train_subset, val_subset = random_split(train_ds, [n_train, n_val], generator=g)

    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def build_model(cfg: TransferPipelineConfig) -> nn.Module:
    """Constrói MobileNetV3-Small, ajusta cabeça e aplica estratégia TL."""
    weights = MobileNet_V3_Small_Weights.DEFAULT if cfg.pretrained else None
    model = mobilenet_v3_small(weights=weights)

    # Ajusta o classificador para CIFAR-10 (10 classes)
    # model.classifier: Sequential([Linear, Hardswish, Dropout, Linear])
    # Substituir última Linear
    last_linear_in = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(last_linear_in, cfg.n_classes)

    # Estratégias:
    # baseline: treina tudo do zero (pretrained=false tipicamente)
    # feature_extraction: congela backbone (features) e treina apenas classifier
    # finetune: congela tudo e descongela últimos N blocos + classifier
    if cfg.strategy == "baseline":
        for p in model.parameters():
            p.requires_grad = True
        return model

    # Congelar tudo inicialmente
    for p in model.parameters():
        p.requires_grad = False

    # Classifier sempre treinável (nova cabeça)
    for p in model.classifier.parameters():
        p.requires_grad = True

    if cfg.strategy == "feature_extraction":
        return model

    # finetune: descongela últimos N blocos do backbone (features é Sequential)
    if cfg.finetune_layers <= 0:
        return model

    features = model.features
    n_blocks = len(features)
    k = min(cfg.finetune_layers, n_blocks)

    # Descongela os últimos k blocos
    for idx in range(n_blocks - k, n_blocks):
        for p in features[idx].parameters():
            p.requires_grad = True

    return model


@torch.no_grad()
def evaluate_model(model: nn.Module, val_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """Avaliação: val_loss e val_acc."""
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


def train_model(
    cfg: TransferPipelineConfig,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, float], list[float], list[float]]:
    """Treino: loga métricas por época e retorna histórico para plot."""
    loss_fn = nn.CrossEntropyLoss()

    # Otimizador apenas com parâmetros treináveis
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_hist: list[float] = []
    val_hist: list[float] = []

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
        val_loss, val_acc = evaluate_model(model, val_loader, device)

        train_hist.append(train_loss)
        val_hist.append(val_loss)

        gap = val_loss - train_loss

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_acc", val_acc, step=epoch)
        mlflow.log_metric("generalization_gap", gap, step=epoch)

        last = {"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc}

    return model, last, train_hist, val_hist


def save_model(
    cfg: TransferPipelineConfig,
    model: nn.Module,
    paths: Dict[str, Path],
    run_id: str,
) -> Dict[str, Path]:
    """Salva modelo e metadados de inferência (mínimo)."""
    model_path = paths["models"] / f"{cfg.name}_{run_id}.pt"
    meta_path = paths["artifacts"] / f"{cfg.name}_{run_id}_meta.json"

    payload = {
        "backbone": cfg.backbone,
        "strategy": cfg.strategy,
        "pretrained": cfg.pretrained,
        "finetune_layers": cfg.finetune_layers,
        "n_classes": cfg.n_classes,
        "image_size": cfg.image_size,
        # Normalização compatível com ImageNet (consolidado)
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "state_dict": None,  # não duplicar no JSON
    }

    torch.save({"state_dict": model.state_dict(), "meta": payload}, model_path)
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {"model_path": model_path, "meta_path": meta_path}


def register_artifacts(
    cfg: TransferPipelineConfig,
    paths: Dict[str, Path],
    artifacts: Dict[str, Path],
    metrics: Dict[str, float],
    run_id: str,
) -> Path:
    """Registra artefatos e relatório no MLflow."""
    mlflow.log_artifact(str(artifacts["model_path"]))
    mlflow.log_artifact(str(artifacts["meta_path"]))

    metrics_path = paths["artifacts"] / f"{cfg.name}_{run_id}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    mlflow.log_artifact(str(metrics_path))

    report_path = paths["reports"] / f"report_{cfg.name}_{run_id}.md"
    report_path.write_text(
        "\n".join(
            [
                f"# {cfg.name}",
                "",
                "## Transfer Learning (pipeline supervisionado de imagens)",
                "",
                "### Estratégia",
                f"- strategy: {cfg.strategy}",
                f"- backbone: {cfg.backbone}",
                f"- pretrained: {cfg.pretrained}",
                f"- finetune_layers: {cfg.finetune_layers}",
                "",
                "### Métricas finais (validação)",
                f"- train_loss: {metrics['train_loss']:.6f}",
                f"- val_loss: {metrics['val_loss']:.6f}",
                f"- val_acc: {metrics['val_acc']:.6f}",
                "",
                "### Parâmetros de treino",
                f"- epochs: {cfg.epochs}",
                f"- batch_size: {cfg.batch_size}",
                f"- lr: {cfg.lr}",
                f"- weight_decay: {cfg.weight_decay}",
                "",
                "### Dados",
                f"- dataset: {cfg.dataset}",
                f"- image_size: {cfg.image_size}",
                f"- val_split: {cfg.val_split}",
                "",
                "### Artefatos",
                f"- model: {artifacts['model_path']}",
                f"- meta: {artifacts['meta_path']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    mlflow.log_artifact(str(report_path))
    return report_path


# ============================================================
# Orquestração
# ============================================================
def run_pipeline(cfg: TransferPipelineConfig) -> None:
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
        # Tags e contexto
        mlflow.set_tag("description", cfg.description or "")
        for k, v in cfg.tags.items():
            mlflow.set_tag(k, v)

        mlflow.log_param("runtime.device", device_str)
        mlflow.log_param("runtime.deterministic", cfg.deterministic)
        mlflow.log_param("runtime.device_preference", cfg.device_preference)

        mlflow.log_param("data.dataset", cfg.dataset)
        mlflow.log_param("data.data_dir", cfg.data_dir)
        mlflow.log_param("data.seed", cfg.seed)
        mlflow.log_param("data.val_split", cfg.val_split)
        mlflow.log_param("data.num_workers", cfg.num_workers)
        mlflow.log_param("data.image_size", cfg.image_size)

        mlflow.log_param("train.epochs", cfg.epochs)
        mlflow.log_param("train.batch_size", cfg.batch_size)
        mlflow.log_param("train.lr", cfg.lr)
        mlflow.log_param("train.weight_decay", cfg.weight_decay)

        mlflow.log_param("model.strategy", cfg.strategy)
        mlflow.log_param("model.backbone", cfg.backbone)
        mlflow.log_param("model.pretrained", cfg.pretrained)
        mlflow.log_param("model.finetune_layers", cfg.finetune_layers)
        mlflow.log_param("model.n_classes", cfg.n_classes)

        mlflow.set_tag("seed.os", seed_report.os)
        mlflow.set_tag("seed.backend", seed_report.backend)
        if seed_report.notes:
            mlflow.set_tag("seed.notes", seed_report.notes)

        # 1) ingestão
        train_ds, _ = ingest_data(cfg)

        # 2) preparação
        train_loader, val_loader = prepare_data(cfg, train_ds)

        # 3) modelo (com estratégia TL)
        model = build_model(cfg).to(device)
        mlflow.log_param("model.trainable_params", count_trainable_params(model))

        # 4) treino
        model, last_metrics, train_hist, val_hist = train_model(cfg, model, train_loader, val_loader, device)

        # 5) salvar modelo
        artifacts = save_model(cfg, model, paths, run_id)

        # 6) registrar artefatos + relatório
        log_loss_curves_as_artifact(cfg, paths, train_hist, val_hist, run_id)
        register_artifacts(cfg, paths, artifacts, last_metrics, run_id)

        print(f"[OK] Transfer pipeline executado: {run_name}")
        print(f"     device={device_str} mlruns={paths['mlruns']}")
