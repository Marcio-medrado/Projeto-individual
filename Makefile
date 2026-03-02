.PHONY: help install mlflow-ui clean exp01 exp02 exp03 exp04 exp06a exp06b exp06c exp07a exp07b exp07c

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
	@echo "  exp04      - roda pipeline completo (incremental via stamp)"
	@echo "  exp06a     - roda variação de early stopping (incremental via stamp)"
	@echo "  exp06b     - roda variação de dropout (incremental via stamp)"
	@echo "  exp06c     - roda variação de weight decay (incremental via stamp)"
	@echo "  exp07a     - roda baseline CNN (incremental via stamp)"
	@echo "  exp07b     - roda feature extraction (incremental via stamp)"
	@echo "  exp07c     - roda fine-tune (incremental via stamp)"
	@echo "  mlflow-ui  - abre MLflow UI (local)"
	@echo "  clean      - remove outputs (models/reports/mlruns/.stamps)"

install:
	cd $(ROOT) && python -m pip install -r requirements.txt
	cd $(ROOT) && python -m pip install -e .

mlflow-ui:
	cd $(ROOT) && mlflow ui --backend-store-uri ./mlruns

clean:
	rm -rf $(ROOT)/models $(ROOT)/artifacts $(ROOT)/reports $(ROOT)/.stamps $(ROOT)/mlruns

exp01: $(STAMP_DIR)/exp01_baseline.ok
exp02: $(STAMP_DIR)/exp02_lr002.ok
exp03: $(STAMP_DIR)/exp03_hidden128.ok
exp04: $(STAMP_DIR)/exp04_pipeline.ok
exp06a: $(STAMP_DIR)/exp06_early_stopping.ok 
exp06b: $(STAMP_DIR)/exp06_dropout.ok 
exp06c: $(STAMP_DIR)/exp06_decay.ok
exp07a: $(STAMP_DIR)/exp07_baseline_cnn.ok
exp07b: $(STAMP_DIR)/exp07_tl_feature_extraction.ok
exp07c: $(STAMP_DIR)/exp07_tl_finetune.ok

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

$(STAMP_DIR)/exp04_pipeline.ok: $(CONFIG_DIR)/exp04_pipeline.yaml $(RUNNER) $(SEEDPY)
	@mkdir -p $(STAMP_DIR)
	cd $(ROOT) && python -m taia_lab.pipelines.run_supervised_pipeline --config configs/exp04_pipeline.yaml
	@touch $@
	
$(STAMP_DIR)/exp06_decay.ok: $(CONFIG_DIR)/exp06_decay_pipeline.yaml $(RUNNER) $(SEEDPY)
	@mkdir -p $(STAMP_DIR)
	cd $(ROOT) && python -m taia_lab.pipelines.run_supervised_pipeline --config configs/exp06_decay_pipeline.yaml
	@touch $@

$(STAMP_DIR)/exp06_early_stopping.ok: $(CONFIG_DIR)/exp06_early_stopping.yaml $(RUNNER) $(SEEDPY)
	@mkdir -p $(STAMP_DIR)
	cd $(ROOT) && python -m taia_lab.pipelines.run_supervised_pipeline --config configs/exp06_early_stopping.yaml
	@touch $@

$(STAMP_DIR)/exp06_dropout.ok: $(CONFIG_DIR)/exp06_dropout_pipeline.yaml $(RUNNER) $(SEEDPY)
	@mkdir -p $(STAMP_DIR)
	cd $(ROOT) && python -m taia_lab.pipelines.run_supervised_pipeline --config configs/exp06_dropout_pipeline.yaml
	@touch $@

$(STAMP_DIR)/exp07_baseline_cnn.ok: $(CONFIG_DIR)/exp07_baseline_cnn.yaml $(RUNNER) $(SEEDPY)
	@mkdir -p $(STAMP_DIR)
	cd $(ROOT) && python -m taia_lab.pipelines.run_transfer_pipeline --config configs/exp07_baseline_cnn.yaml
	@touch $@

$(STAMP_DIR)/exp07_tl_feature_extraction.ok: $(CONFIG_DIR)/exp07_tl_feature_extraction.yaml $(RUNNER) $(SEEDPY)
	@mkdir -p $(STAMP_DIR)
	cd $(ROOT) && python -m taia_lab.pipelines.run_transfer_pipeline --config configs/exp07_tl_feature_extraction.yaml
	@touch $@

$(STAMP_DIR)/exp07_tl_finetune.ok: $(CONFIG_DIR)/exp07_tl_finetune.yaml $(RUNNER) $(SEEDPY)
	@mkdir -p $(STAMP_DIR)
	cd $(ROOT) && python -m taia_lab.pipelines.run_transfer_pipeline --config configs/exp07_tl_finetune.yaml
	@touch $@

