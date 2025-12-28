# ⚽ BRMP — Brazilian Match Prediction

**Previsão de resultados do futebol brasileiro com Machine Learning**

Projeto end-to-end de ciência de dados aplicado ao futebol brasileiro, desenvolvendo modelos preditivos a partir de dados históricos, estatísticas avançadas e engenharia de features robusta.

---

## 🎯 Objetivo

Construir um pipeline completo e reproduzível de análise preditiva capaz de:

- **Coletar e estruturar** dados históricos de partidas do Brasileirão
- **Gerar features estatísticas** avançadas (forma recente, desempenho casa/fora, saldo de gols, sequências)
- **Treinar modelos baseline** calibrados e interpretáveis
- **Avaliar performance** com métricas rigorosas (accuracy, log loss, Brier score)
- **Estabelecer fundação** para otimizações futuras (feature selection, ensembles, modelos temporais)

---

## 🧠 Abordagem Metodológica

O projeto segue boas práticas de Data Science e MLOps, com foco em **reprodutibilidade** e **escalabilidade**:

1. **Data Collection & Cleaning** — Coleta e tratamento de dados históricos
2. **Feature Engineering** — Criação de variáveis preditivas relevantes
3. **Feature Freeze** — Versionamento e congelamento dos datasets para experimentos consistentes
4. **Baseline Modeling** — Desenvolvimento de modelos calibrados como referência
5. **Validation & Evaluation** — Validação temporal e análise de performance
6. **Iteration** — Ciclo de melhoria contínua com novos modelos e features

---

## 📁 Estrutura do Projeto
```
BRMP-Brazilian-Match-Prediction/
│
├── data/
│   ├── raw/                     # Dados brutos coletados (CSV, JSON)
│   ├── processed/               # Dados limpos e estruturados
│   └── features/                # Feature sets versionados
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_modeling.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/
│   ├── data_processing.py       # Pipeline de limpeza e transformação
│   ├── feature_engineering.py   # Criação de features estatísticas
│   ├── modeling.py              # Treinamento e calibração de modelos
│   └── evaluation.py            # Métricas e validação
│
├── models/                      # Modelos treinados (.pkl, .joblib)
│
├── reports/
│   ├── figures/                 # Gráficos e visualizações
│   └── results.md               # Resumo de experimentos
│
├── tests/                       # Testes unitários
│
├── .gitignore
├── README.md
├── requirements.txt
└── setup.py
```

---

## 🧪 Modelagem

### 🔹 Baseline Model

Modelo inicial calibrado para servir como **referência quantitativa** em comparações futuras:

- **Algoritmo:** Logistic Regression / Random Forest (calibrado)
- **Métricas de avaliação:**
  - **Accuracy** — Taxa de acerto geral
  - **Log Loss** — Qualidade das probabilidades previstas
  - **Brier Score** — Calibração probabilística
  - **ROC-AUC** — Discriminação entre classes

### 🔹 Objetivo do Baseline

Estabelecer um **ponto de comparação robusto** para modelos mais sofisticados no futuro:
- Gradient Boosting (LightGBM, XGBoost, CatBoost)
- Redes neurais (LSTM para séries temporais)
- Ensembles e stacking

---

## 📊 Status Atual

| Etapa                            | Status          |
|----------------------------------|-----------------|
| Estrutura inicial do projeto     | ✅ Completo     |
| Coleta e organização dos dados   | ✅ Completo     |
| Feature engineering básico       | ✅ Completo     |
| Baseline model calibrado         | ✅ Completo     |
| Documentação (README)            | ✅ Completo     |
| Feature engineering avançado     | 🟡 Em progresso |
| Validação temporal (time-split)  | 🟡 Em progresso |
| Ajuste de hiperparâmetros        | 🔜 Próximo      |
| Análise de importância (SHAP)    | 🔜 Próximo      |

---

## 🚀 Roadmap

### **Fase 1: Consolidação** (Atual)
- [ ] Validação temporal com rolling windows
- [ ] Feature selection com importância e correlação
- [ ] Ajuste fino de hiperparâmetros (Grid/Random Search)

### **Fase 2: Otimização**
- [ ] Modelos avançados (XGBoost, LightGBM, CatBoost)
- [ ] Ensemble learning (voting, stacking)
- [ ] Análise de interpretabilidade (SHAP, LIME)

### **Fase 3: Produção**
- [ ] API para inferência (FastAPI)
- [ ] Dashboard interativo (Streamlit)
- [ ] Monitoramento de performance em produção

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.9+**
- **Pandas, NumPy** — Manipulação de dados
- **Scikit-learn** — Modelagem e validação
- **XGBoost, LightGBM** — Gradient boosting
- **Matplotlib, Seaborn** — Visualização
- **Jupyter Notebook** — Experimentação

---

## 📦 Instalação
```bash
# Clone o repositório
git clone https://github.com/btaguiar/BRMP-Brazilian-Match-Prediction.git
cd BRMP-Brazilian-Match-Prediction

# Crie um ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instale as dependências
pip install -r requirements.txt
```

---

## 🧑‍💻 Autor

**Bruno Aguiar**  
Data Analytics & Machine Learning

🔗 [GitHub](https://github.com/btaguiar)  
🔗 [LinkedIn](https://www.linkedin.com/in/bruno-aguiar-marketing-analytics/)

---

## ⚠️ Disclaimer

Este projeto tem **fins exclusivamente educacionais e analíticos**. Não possui objetivo comercial nem incentiva apostas esportivas.

---

## 🤝 Contribuições

Sugestões, melhorias e pull requests são **muito bem-vindos**!

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

---

## 📄 Licença

Este projeto está sob a licença MIT. Consulte o arquivo `LICENSE` para mais detalhes.
