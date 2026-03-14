# 🔬 Análise Estatística Avançada e Dashboard E-Commerce (Olist)

## 👥 Autores
Projeto desenvolvido e arquitetado por:
* **Anderson Marcolino**
* **Alessandra Brasiliano**
* **Marineide Gomes**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-red)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Data Science](https://img.shields.io/badge/Data%20Science-Estatística%20Avançada-success)

Este projeto representa o estado da arte em análise de dados End-to-End, utilizando o ecossistema completo de dados públicos do e-commerce brasileiro **Olist** (9 bases de dados interligadas). O sistema contempla um pipeline robusto de Engenharia de Dados (ETL) e uma aplicação interativa que une Estatística Descritiva, Inteligência Geoespacial, Segmentação de Marketing e Machine Learning.

## 🎯 O Que Resolvemos?
Transformamos mais de 100 mil registros brutos de pedidos, pagamentos, geolocalização e avaliações em um motor de inteligência de negócios. A ferramenta permite desde a compreensão básica da distribuição das vendas até a previsão algorítmica de tempos de entrega.

## 🌟 Arquitetura e Funcionalidades (As 6 Abas)
O painel Streamlit é totalmente responsivo a um **Filtro Global por Estado (UF)** e está dividido em 6 visões estratégicas:

1. **📊 Estatística Básica:** Dissecação de variáveis contínuas (Frete, Preço, Distância). Cálculo de medidas de tendência central (Média, Mediana, Moda), dispersão (Desvio Padrão) e assimetria (Skewness). Inclui visualizações duplas (Histogramas + Boxplots) para detecção de *outliers*.
2. **🗺️ Mapa & Regional:** Inteligência geoespacial utilizando mapas de calor (Densidade) baseados na latitude e longitude exata de clientes e vendedores.
3. **📈 Estatística Avançada:** Validação matemática de hipóteses de negócio. Inclui regressão linear simples (Correlação de Pearson) e Testes de Hipótese Dinâmicos (Teste T de Student / A/B Testing) para comparar a performance logística entre estados.
4. **👥 Segmentação RFM:** Motor de marketing que classifica automaticamente os clientes em clusters baseados em Recência, Frequência e Valor Monetário (ex: "🌟 Campeões", "⚠️ Risco de Perda").
5. **🤖 Previsão (Machine Learning):** Um modelo preditivo treinado em tempo real (Regressão Linear Múltipla via `scikit-learn`) que prevê o tempo de entrega de um novo pedido com base na distância em quilômetros, valor do frete e preço do produto.
6. **🗄️ Explorador de Dados (Dataset):** Transparência total. Visualização em formato de grade interativa da Tabela Analítica de Dados (ABT) com a funcionalidade de **Download (CSV)** em tempo real dos dados já com os filtros globais aplicados.

## ⚙️ O Pipeline de ETL (Under the Hood)
O script `etl_estatistico.py` não faz apenas *joins*. Ele aplica inteligência:
* **Fórmula de Haversine:** Calcula a distância geodésica real (em KM) entre o CEP do cliente e do vendedor usando trigonometria.
* **Resolução de Cardinalidade:** Agrupa múltiplos métodos de pagamento por pedido para evitar duplicatas na base.
* **Alta Performance:** Exporta o resultado final em formato `.parquet`, garantindo leitura ultrarrápida na aplicação web.
