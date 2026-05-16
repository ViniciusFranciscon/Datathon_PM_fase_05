# 🎓 Datathon PM · Diagnóstico Preditivo de Risco Educacional

## 📌 Visão Geral

Projeto desenvolvido para o **Datathon FIAP · Pós Tech** com foco na construção de uma solução analítica preditiva para apoio à tomada de decisão educacional da **Associação Passos Mágicos**.

A proposta do projeto consiste na utilização de técnicas de:

* Machine Learning;
* Clusterização;
* Explainable AI (XAI);
* Business Intelligence;
* Data Storytelling;

para identificar antecipadamente alunos com maior probabilidade de defasagem educacional e permitir intervenções pedagógicas direcionadas.

---

# 🚀 Objetivo do Projeto

Desenvolver uma solução inteligente capaz de:

✅ Identificar alunos com risco de deterioração educacional;

✅ Antecipar possíveis quedas de desempenho;

✅ Explicar os fatores responsáveis pelo risco previsto;

✅ Classificar perfis comportamentais e educacionais;

✅ Apoiar a priorização de intervenções pedagógicas;

✅ Disponibilizar uma interface executiva interativa para análise individual.

---

# 🏛️ Contexto Institucional

A **Associação Passos Mágicos** atua há mais de 30 anos promovendo transformação social por meio da educação para crianças e jovens em situação de vulnerabilidade social.

O projeto busca apoiar a instituição através de uma abordagem orientada por dados, permitindo maior eficiência na identificação de alunos que demandam atenção prioritária.

---

# 🧠 Arquitetura Analítica da Solução

```text
Base Educacional Histórica (2022–2024)
                ↓
     Análise Exploratória de Dados
                ↓
       Engenharia de Features
                ↓
      Modelo Preditivo XGBoost
                ↓
        Clusterização KMeans
                ↓
         Explainability SHAP
                ↓
       Dashboard em Streamlit
                ↓
 Diagnóstico Executivo Individual
```

---

# 🔬 Técnicas e Modelos Utilizados

## 📈 Modelo Preditivo — XGBoost

O modelo principal da solução foi desenvolvido utilizando o algoritmo **XGBoost Classifier**, escolhido devido à:

* alta performance preditiva;
* robustez em bases tabulares;
* capacidade de generalização;
* eficiência computacional.

O objetivo do modelo é prever a probabilidade de risco educacional com base nos indicadores institucionais.

---

## 🧩 Clusterização — KMeans

Foi implementado um modelo de clusterização utilizando **KMeans** para segmentar alunos em perfis educacionais semelhantes.

Os clusters identificam padrões como:

* deterioração gradual;
* risco estrutural;
* falha pontual de aprendizado;
* queda recente de desempenho.

Essa segmentação permite análises mais direcionadas e estratégicas.

---

## 🔍 Explainable AI — SHAP

Para aumentar a interpretabilidade do modelo, foi utilizada a biblioteca **SHAP (SHapley Additive exPlanations)**.

A solução permite identificar:

* quais indicadores mais contribuíram para o risco previsto;
* quais fatores reduziram o risco;
* impacto individual de cada variável.

Isso garante maior transparência analítica e apoio à tomada de decisão.

---

# 📊 Indicadores Utilizados

Os principais indicadores educacionais utilizados foram:

| Indicador | Descrição                         |
| --------- | --------------------------------- |
| IDA       | Indicador de Desempenho Acadêmico |
| IEG       | Indicador de Engajamento          |
| IPS       | Indicador Psicossocial            |
| IPP       | Indicador Psicopedagógico         |
| IPV       | Indicador de Ponto de Virada      |
| IAN       | Indicador de Adequação de Nível   |
| IAA       | Indicador de Autoavaliação        |

---

# 🖥️ Dashboard Interativo

O projeto disponibiliza um dashboard executivo desenvolvido em **Streamlit**, permitindo:

✅ Inserção manual dos indicadores do aluno;

✅ Geração automática de diagnóstico preditivo;

✅ Visualização do risco composto;

✅ Análise visual comparativa;

✅ Interpretação SHAP;

✅ Simulações What-If;

✅ Recomendações de intervenção.

---

# 🌐 Aplicação Online

## 🔗 Streamlit Community Cloud

Acesse a aplicação online:

👉 [https://datathonpmfase05-uv4gj34rqee4w4b8jkn9jq.streamlit.app/](https://datathonpmfase05-uv4gj34rqee4w4b8jkn9jq.streamlit.app/)

---

# 📂 Repositório GitHub

## 🔗 GitHub

Acesse o repositório completo:

👉 [https://github.com/ViniciusFranciscon/Datathon_PM_fase_05](https://github.com/ViniciusFranciscon/Datathon_PM_fase_05)

---

# 📁 Estrutura do Projeto

```text
Datathon_PM_fase_05/
│
├── app/
│   └── app.py
│
├── data/
│   └── df_model.csv
│
├── data_raw/
│   └── base_datathon_pm_vf.csv
│
├── img/
│
├── notebooks/
│   ├── notebook_modelagem_datathon_fase_05.ipynb
│   └── notebook_datathon_fase_05_q&a.ipynb
│
├── pkl/
│   ├── modelo.pkl
│   ├── scaler.pkl
│   ├── cluster_model.pkl
│   ├── cluster_scaler.pkl
│   └── shap_values.pkl
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

# 📓 Notebooks Disponíveis

## 📘 Análise Exploratória e Modelagem

Notebook responsável por:

* análise exploratória;
* tratamento dos dados;
* feature engineering;
* treinamento dos modelos;
* avaliação dos resultados;
* exportação dos artefatos `.pkl`.

---

## 📗 Notebook Q&A

Notebook complementar contendo:

* análises adicionais;
* respostas estratégicas;
* exploração analítica do problema de negócio.

---

# ⚙️ Como Executar o Projeto Localmente

## 1️⃣ Clone o repositório

```bash
git clone https://github.com/ViniciusFranciscon/Datathon_PM_fase_05.git
```

---

## 2️⃣ Acesse a pasta do projeto

```bash
cd Datathon_PM_fase_05
```

---

## 3️⃣ Instale as dependências

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Execute o Streamlit

```bash
streamlit run app/app.py
```

---

# 📦 Principais Bibliotecas Utilizadas

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* SHAP
* Streamlit
* Plotly
* Matplotlib
* Seaborn
* Pillow
* Joblib

---

# 📈 Principais Entregas do Projeto

## ✅ Modelo preditivo funcional

## ✅ Dashboard executivo interativo

## ✅ Clusterização comportamental

## ✅ Explainable AI com SHAP

## ✅ Simulador What-If

## ✅ Deploy em produção no Streamlit Cloud

## ✅ Versionamento completo no GitHub

---

# 💡 Diferenciais da Solução

✔️ Explicabilidade do modelo;

✔️ Interface executiva amigável;

✔️ Arquitetura reprodutível;

✔️ Integração entre negócio e analytics;

✔️ Aplicação prática de IA para impacto social;

✔️ Tomada de decisão orientada por dados.

---

# 📊 Resultado Esperado

A solução permite:

* aumentar a capacidade de monitoramento educacional;
* identificar alunos prioritários;
* reduzir riscos de deterioração educacional;
* apoiar decisões pedagógicas estratégicas;
* transformar dados em ações direcionadas.

---

# 👨‍💻 Autor

## Vinicius Franciscon

Projeto desenvolvido para o Datathon FIAP · Pós Tech.

---

# 🏆 Considerações Finais

Este projeto representa a aplicação integrada de:

* ciência de dados;
* machine learning;
* visualização executiva;
* inteligência analítica;
* explainable AI;
* deploy em nuvem;

com foco na geração de impacto educacional e social.

A solução demonstra como modelos preditivos podem ser utilizados não apenas para previsão, mas também para apoio estratégico à tomada de decisão em ambientes de alta relevância social.
