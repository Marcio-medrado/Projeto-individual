from __future__ import annotations

import os
import platform
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@dataclass(frozen=True)
class SeedReport:
    """Relatório mínimo para auditoria de reprodutibilidade (útil em MLOps/TAIA)."""
    seed: int
    os: str
    backend: str
    device: str
    deterministic_requested: bool
    deterministic_effective: bool
    notes: str = ""


def _os_name() -> str:
    # Linux / Windows / Darwin (macOS)
    return platform.system()


def prefer_device(prefer: str = "auto") -> str:
    """Escolhe um device coerente com o ambiente.

    prefer:
      - "auto": cuda > mps > cpu (quando disponível)
      - "cuda": força cuda se disponível, senão cai para cpu
      - "mps": força mps se disponível, senão cai para cpu
      - "cpu": força cpu
    """
    if torch is None:
        return "cpu"

    prefer = (prefer or "auto").lower().strip()

    has_cuda = torch.cuda.is_available()
    has_mps = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()

    if prefer == "cuda":
        return "cuda" if has_cuda else "cpu"
    if prefer == "mps":
        return "mps" if has_mps else "cpu"
    if prefer == "cpu":
        return "cpu"

    # auto
    if has_cuda:
        return "cuda"
    if has_mps:
        return "mps"
    return "cpu"


def seed_everything(
    seed: int = 42,
    *,
    deterministic: bool = True,
    device_preference: str = "auto",
    set_pythonhashseed: bool = True,
) -> SeedReport:
    """Define seeds e tenta configurar determinismo de forma multi-plataforma.

    - Linux/Windows: funciona bem com CPU/CUDA (com limitações típicas do PyTorch).
    - macOS (Apple Silicon): suporta MPS; determinismo pode ser parcial dependendo da versão/ops.

    Retorna um SeedReport (bom para log em MLflow).
    """
    os_name = _os_name()

    # 1) Seeds básicos (válidos em qualquer OS)
    if set_pythonhashseed:
        # Ajuda em operações com hashing e ordenação de dict/set em alguns cenários.
        os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    if torch is None:
        return SeedReport(
            seed=seed,
            os=os_name,
            backend="no_torch",
            device="cpu",
            deterministic_requested=deterministic,
            deterministic_effective=False,
            notes="torch não está instalado; apenas random/numpy foram semeados.",
        )

    # 2) Torch seeds (CPU + aceleradores)
    torch.manual_seed(seed)

    # CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # MPS (Apple Silicon)
    # Não há manual_seed específico para MPS: torch.manual_seed cobre geradores.
    # Mas o determinismo pode não ser garantido (depende de operador/versão).
    has_mps = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()

    # 3) Configurações de determinismo (melhor esforço)
    deterministic_effective = False
    notes = ""

    if deterministic:
        # CuDNN determinism (impacta principalmente CUDA)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # PyTorch global determinism (quando suportado)
        # Observação: pode lançar erro se operações não determinísticas forem usadas.
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                # warn_only=True reduz fricção didática; muda para False se quiser "falhar rápido".
                torch.use_deterministic_algorithms(True, warn_only=True)
                deterministic_effective = True
            except Exception as e:  # pragma: no cover
                deterministic_effective = False
                notes += f"use_deterministic_algorithms falhou: {e}. "

        # CUDA: alguns kernels exigem env vars para determinismo
        if torch.cuda.is_available():
            # Recomendação do PyTorch para determinismo em GEMM/CUBLAS
            # (não quebra em Windows/Linux; pode ser ignorado se não for aplicável).
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
            notes += "CUDA detectado; CUBLAS_WORKSPACE_CONFIG configurado. "

        # MPS: determinismo é limitado
        if os_name == "Darwin" and has_mps:
            notes += (
                "MPS (Apple Silicon) detectado: determinismo pode ser parcial dependendo "
                "da versão do PyTorch e dos operadores usados. "
            )
            # Em alguns casos, forçar fallback para CPU reduz variância, mas custa performance.
            # Não aplicamos automaticamente para não surpreender o usuário.
    else:
        # Permite performance máxima (especialmente CUDA), mas resultados podem variar.
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True
        deterministic_effective = False
        notes += "deterministic=False (maior performance; menor reprodutibilidade). "

    device = prefer_device(device_preference)
    backend = "cuda" if device == "cuda" else ("mps" if device == "mps" else "cpu")

    # Ajuste: mesmo com deterministic=True, em MPS não garantimos determinismo "efetivo"
    if backend == "mps" and deterministic:
        # Mantemos deterministic_effective como True apenas se use_deterministic_algorithms foi aplicado,
        # mas deixamos a nota explícita. Isso evita falsa promessa.
        pass

    return SeedReport(
        seed=seed,
        os=os_name,
        backend=backend,
        device=device,
        deterministic_requested=deterministic,
        deterministic_effective=deterministic_effective,
        notes=notes.strip(),
    )


def set_default_dtype(dtype: str = "float32") -> None:
    """Conveniente para padronizar dtype em ambientes heterogêneos."""
    if torch is None:
        return
    dtype = dtype.lower().strip()
    mapping = {
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
        "fp64": torch.float64,
    }
    if dtype in mapping:
        torch.set_default_dtype(mapping[dtype])