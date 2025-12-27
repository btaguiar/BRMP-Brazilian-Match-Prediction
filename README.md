# ⚽ BRMP — Brazilian Match Prediction

Projeto de **Machine Learning aplicado ao futebol** com foco na **previsão de resultados do Campeonato Brasileiro (Série A)**, utilizando dados históricos, engenharia de variáveis temporais e validação rigorosa entre múltiplas fontes.

---

## 🎯 Objetivo

Desenvolver um pipeline completo de dados e modelagem capaz de estimar **probabilidades de vitória do mandante, empate ou vitória do visitante**, respeitando a ordem temporal dos dados e evitando vazamento de informação (*data leakage*).

O projeto foi pensado como um **case realista de Data Science / Machine Learning**, indo da ingestão de dados até a simulação de temporadas futuras.

---

## 📊 Fontes de Dados

O projeto utiliza duas fontes independentes:

- **CBF (via Base dos Dados)**  
  Utilizada para análise exploratória e validação histórica (2003–2024)

- **football-data.co.uk**  
  Fonte principal do pipeline de modelagem (2012–2025)

> ⚠️ Os datasets não são versionados neste repositório.  
> O projeto documenta como os dados são obtidos, limpos e processados localmente.

---

## 🧪 Validação entre Fontes

As duas bases foram comparadas no período **2012–2023**, com os seguintes resultados:

- Cobertura anual idêntica (380 jogos/ano)
- Placares **100% consistentes**
- Divergências residuais apenas na **data de registro** (±1 dia)
- Cobertura total com tolerância temporal: **99,82%**

📌 **Decisão de arquitetura:**  
O pipeline adota *football-data.co.uk* como fonte principal por maior consistência temporal.  
A base da CBF é utilizada para validação cruzada e histórico estendido.

Detalhes completos no notebook:  
`notebooks/02_compare_sources.ipynb`

---

## 🧠 Engenharia de Variáveis

O modelo não utiliza apenas o placar final. Entre as principais features:

- 📈 **Médias móveis dos últimos jogos**
- ⚔️ **Força de ataque e defesa**
- 🏠 **Vantagem do mando de campo**
- 🔁 **Separação clara entre desempenho como mandante e visitante**
- ⏱️ **Ordenação temporal estrita (sem data leakage)**

---

## 🧪 Metodologia de Treino e Avaliação

- **Treino:** 2012–2023  
- **Teste:** 2024  
- **Contexto Atual:** 2025  
- **Simulação:** Temporada 2026

As métricas utilizadas são probabilísticas, adequadas para problemas de previsão esportiva:

- Log Loss  
- Brier Score  

---

## 🗂️ Estrutura do Projeto

```text
├── notebooks/        # EDA, validações e preparação dos dados
├── src/              # Código reutilizável
│   ├── data/         # Ingestão, limpeza e features
│   ├── models/       # Treino, avaliação e predição
│   └── utils/        # Métricas, validações e helpers
├── tests/            # Testes automatizados
├── docs/             # Documentação técnica
└── data/             # Dados locais (não versionados)
```

## 🚀 Próximos Passos
- Implementação do modelo de classificação (LogReg, XGBoost)
- Avaliação por Log Loss e Brier Score
- Simulação completa da temporada 2026
- Integração futura com APIs de jogos em tempo real

## 👤 Autor
Bruno Aguiar  
Marketing Analytics • Data Science • Football Analytics  

GitHub: https://github.com/btaguiar  
LinkedIn: https://www.linkedin.com/in/bruno-aguiar-marketing-analytics/
