# TAIA Lab — Experimentos, Métricas e MLOps

Repositorio-base da disciplina TAIA.

- **código** (pipeline) vs. **experimento** (configuração + hipótese + métricas + rastreabilidade);
- **execução local** vs. **trabalho sustentável** (MLOps).

## Estrutura
- `taia_lab/pipelines/`
  - `minimal_pipeline.py` (Aula 01)
  - `tracked_pipeline.py` (Aula 02 — MLflow local)
  - `run_experiment.py` (Aula 03 — YAML + MLflow)
- `configs/` — especificações declarativas dos experimentos (YAML)
- `models/` — artefatos gerados (saída)
- `reports/` — relatórios gerados (saída)
- `mlruns/` — tracking local do MLflow (saída)
- `docs/` — textos e diagramas de apoio

## Rodando um experimento (Aula 03)
```bash
pip install -r requirements.txt
pip install -e .

python -m taia_lab.pipelines.run_experiment --config configs/exp01_baseline.yaml
python -m taia_lab.pipelines.run_experiment --config configs/exp02_lr002.yaml
python -m taia_lab.pipelines.run_experiment --config configs/exp03_hidden128.yaml
```

## Visualizando o tracking (MLflow local)
```bash
mlflow ui --backend-store-uri ./mlruns
```
## Observação importante

Este laboratório usa dados sintéticos (make_classification) para reduzir fricção didática.
Em aulas seguintes, o pipeline pode ser estendido para dados reais e validações adicionais.