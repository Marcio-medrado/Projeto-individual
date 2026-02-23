from pathlib import Path

from __future__ import annotations

def project_root() -> Path:
    """Detecta a raiz do projeto (presença de pyproject.toml ou .git)."""
    p = Path(__file__).resolve()
    for parent in [p, *p.parents]:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    raise RuntimeError("Não foi possível detectar a raiz do projeto (pyproject.toml/.git).")