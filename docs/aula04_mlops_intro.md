# Aula 04 — Introdução ao MLOps (TAIA)

## Ponto de partida (conexão com a aula anterior)

Na aula anterior, vocês já praticaram três pilares de MLOps em nível experimental:
- **Especificação declarativa** do experimento (YAML)
- **Métricas** para comparar hipóteses (val_loss, val_acc)
- **Rastreabilidade** por tracking (MLflow)

A proposta desta aula é ampliar a visão: **um experimento só vira engenharia quando ele é sustentado no tempo**.

---

## Objetivos de aprendizagem
Ao final da aula, você deve ser capaz de:
1. Explicar **por que MLOps existe** (organização do trabalho).
2. Representar o **ciclo de vida do modelo** de forma simples.
3. Identificar quais **artefatos** precisam ser registrados para sustentar uma decisão (dados, código, config, modelo, métricas).
4. Relacionar a prática com YAML/MLflow ao conceito de **registro de modelos** e **noção de deploy**.

---

## Conteúdo (sequência sugerida)
### 1) O problema real
Cenário: “o Experimento 3 ficou melhor e vamos colocar em uso”.
Perguntas que quebram a intuição “métrica resolve tudo”:
- Qual dataset (versão/recorte) foi usado?
- Qual versão do código gerou o modelo?
- Quais métricas decidiram a troca do baseline?
- Como você volta ao mesmo resultado daqui a 2 meses?

**Tese da aula:** MLOps torna explícitas as dependências e registra o contexto de decisão.

### 2) Ciclo de vida do modelo (visão mínima, mas correta)
- Dados → Treino → Avaliação → Modelo → Uso (deploy) → Observação → Reavaliação/Atualização
- “Deploy” não é fim; é **início de monitoramento**.
- Obsolescência: drift, mudança de objetivos, mudança de dados.

### 3) O que precisa ser rastreado
- **Dados**: origem, recorte, versão
- **Código**: commit/tag
- **Config**: YAML do experimento
- **Modelo**: pesos + metadados
- **Métricas**: treino/validação + critérios de decisão
- **Decisão**: justificativa para usar/atualizar/substituir

### 4) Registro de modelos (conceito)
Um “registro” é o catálogo de modelos candidatos/ativos com:
- referência ao experimento que gerou o modelo
- métricas associadas
- artefatos (pesos/relatórios)
- status (candidato, aprovado, em produção, aposentado)

