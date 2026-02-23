.PHONY: help install mlflow-ui clean exp01 exp02 exp03

ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
STAMP_DIR := $(ROOT)/.stamps
CONFIG_DIR := $(ROOT)/configs
RUNNER := $(ROOT)/src/taia_lab/pipelines/run_experiment.py
SEEDPY := $(ROOT)/src/taia_lab/utils/seed.py

help:
	@echo "Targets:"
	@echo "  install    - instala dependências e pacote em modo editável"
	@echo "  exp01      - roda baseline (incremental via stamp)"
	@echo "  exp02      - roda variação de lr (incremental via stamp)"
	@echo "  exp03      - roda hidden_dim=128 (incremental via stamp)"
	@echo "  mlflow-ui  - abre MLflow UI (local)"
	@echo "  clean      - remove outputs (models/reports/mlruns/.stamps)"

install:
	cd $(ROOT) && python -m pip install -r requirements.txt
	cd $(ROOT) && python -m pip install -e .

mlflow-ui:
	cd $(ROOT) && mlflow ui --backend-store-uri ./mlruns

clean:
	rm -rf $(ROOT)/models $(ROOT)/reports $(ROOT)/mlruns $(ROOT)/.stamps

exp01: $(STAMP_DIR)/exp01_baseline.ok
exp02: $(STAMP_DIR)/exp02_lr002.ok
exp03: $(STAMP_DIR)/exp03_hidden128.ok

$(STAMP_DIR)/exp01_baseline.ok: $(CONFIG_DIR)/exp01_baseline.yaml $(RUNNER) $(SEEDPY)
	@mkdir -p $(STAMP_DIR)
	cd $(ROOT) && python -m taia_lab.pipelines.run_experiment --config configs/exp01_baseline.yaml
	@touch $@

$(STAMP_DIR)/exp02_lr002.ok: $(CONFIG_DIR)/exp02_lr002.yaml $(RUNNER) $(SEEDPY)
	@mkdir -p $(STAMP_DIR)
	cd $(ROOT) && python -m taia_lab.pipelines.run_experiment --config configs/exp02_lr002.yaml
	@touch $@

$(STAMP_DIR)/exp03_hidden128.ok: $(CONFIG_DIR)/exp03_hidden128.yaml $(RUNNER) $(SEEDPY)
	@mkdir -p $(STAMP_DIR)
	cd $(ROOT) && python -m taia_lab.pipelines.run_experiment --config configs/exp03_hidden128.yaml
	@touch $@
