# PRD — Ajuste de Capacidade do Modelo

## Problema
O modelo apresenta desempenho limitado na classificação, possivelmente devido à baixa capacidade representacional.

## Configuração base
configs/exp04_pipeline.yaml

## Alteração proposta
Aumentar o parâmetro hidden_dim do modelo de 64 para 128 para avaliar o impacto na acurácia. O experimento deve ser retratado em um novo arquivo de configuração: configs/exp04.1_pipeline.yaml

## Evidência esperada
Comparação entre:
- acurácia do modelo antes e depois da alteração;
- comportamento da loss durante o treinamento;
- diferença registrada no MLflow.