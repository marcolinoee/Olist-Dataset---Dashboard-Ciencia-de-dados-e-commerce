import streamlit as st
import pandas as pd
import plotly.express as px
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
import numpy as np

# --- CONFIGURAÇÃO INICIAL ---
st.set_page_config(page_title="Dashboard 3.0 - Olist Analytics", layout="wide", page_icon="🚀")

@st.cache_data
def carregar_dados():
    return pd.read_parquet('dados_estatisticos.parquet')

@st.cache_resource
def treinar_modelo_previsao(df):
    """Treina um modelo de Machine Learning rápido para prever o tempo de entrega."""
    features = ['freight_value', 'distancia_logistica_km', 'price']
    df_ml = df.dropna(subset=features + ['tempo_entrega_dias']).copy()
    if len(df_ml) < 10:
        return None, None, 0
    
    X = df_ml[features]
    y = df_ml['tempo_entrega_dias']
    
    modelo = LinearRegression()
    modelo.fit(X, y)
    score = modelo.score(X, y)
    return modelo, features, score

def main():
    st.title("🚀 Olist Analytics 3.0: O Dashboard Definitivo")
    
    try:
        df = carregar_dados()
    except FileNotFoundError:
        st.error("Arquivo 'dados_estatisticos.parquet' não encontrado. Rode o script de ETL primeiro!")
        st.stop()

    # --- BARRA LATERAL: FILTRO GLOBAL ---
    st.sidebar.header("🌍 Filtros Globais")
    estados_disponiveis = sorted(df['customer_state'].unique())
    estados_selecionados = st.sidebar.multiselect("Filtre por Estado(s):", estados_disponiveis)

    if estados_selecionados:
        df_filtrado = df[df['customer_state'].isin(estados_selecionados)]
    else:
        df_filtrado = df

    if df_filtrado.empty:
        st.warning("Nenhum dado encontrado com os filtros atuais.")
        st.stop()

    # --- ABAS DA APLICAÇÃO ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Estatística Básica", 
        "🗺️ Mapa & Regional", 
        "📈 Estatística Avançada",
        "👥 Segmentação RFM", 
        "🤖 Previsão (ML)",
        "🗄️ Dados Brutos" 
    ])

    # --- ABA 1: ESTATÍSTICA BÁSICA ---
    with tab1:
        st.header("Análise Descritiva Profunda")
        st.markdown("Mergulho nas distribuições, tendências centrais e dispersão dos dados.")
        
        # Seleção da variável
        variaveis_num = {
            'Valor Total do Item (R$)': 'valor_total_item',
            'Valor do Frete (R$)': 'freight_value',
            'Tempo de Entrega (Dias)': 'tempo_entrega_dias',
            'Distância Logística (KM)': 'distancia_logistica_km',
            'Preço do Produto (R$)': 'price'
        }
        nome_var = st.selectbox("Selecione a métrica para dissecar:", list(variaveis_num.keys()))
        col_var = variaveis_num[nome_var]
        
        # Cálculos Estatísticos
        serie = df_filtrado[col_var].dropna()
        media = serie.mean()
        mediana = serie.median()
        moda = serie.mode().iloc[0] if not serie.mode().empty else np.nan
        desvio = serie.std()
        assimetria = serie.skew()
        
        # Exibição de Métricas (KPIs)
        st.subheader("Medidas de Tendência Central e Dispersão")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Média", f"{media:,.2f}")
        c2.metric("Mediana", f"{mediana:,.2f}")
        c3.metric("Moda", f"{moda:,.2f}")
        c4.metric("Desvio Padrão (σ)", f"{desvio:,.2f}")
        c5.metric("Assimetria (Skew)", f"{assimetria:,.2f}")
        
        st.divider()
        
        # Gráficos de Distribuição
        col_graf, col_tab = st.columns([2, 1])
        with col_graf:
            fig_dist = px.histogram(
                df_filtrado, x=col_var, nbins=60, 
                marginal="box", # Adiciona o Boxplot em cima do Histograma!
                title=f"Distribuição e Outliers: {nome_var}",
                color_discrete_sequence=['#4B0082']
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
        with col_tab:
            st.subheader("Resumo (Describe)")
            st.dataframe(serie.describe(), use_container_width=True)

    # --- ABA 2: MAPA & REGIONAL ---
    with tab2:
        st.header("Visão Geoespacial")
        
        df_mapa = df_filtrado.dropna(subset=['cliente_lat', 'cliente_lng']).sample(min(10000, len(df_filtrado)), random_state=42)
        fig_mapa = px.density_mapbox(
            df_mapa, lat='cliente_lat', lon='cliente_lng', z='valor_total_item', 
            radius=8, center=dict(lat=-15.78, lon=-47.92), zoom=3,
            mapbox_style="open-street-map", title="Densidade de Faturamento"
        )
        st.plotly_chart(fig_mapa, use_container_width=True)

    # --- ABA 3: ESTATÍSTICA AVANÇADA ---
    with tab3:
        st.header("Correlação (Pearson)")
        df_sample = df_filtrado.sample(min(5000, len(df_filtrado)), random_state=42)
        fig_corr = px.scatter(
            df_sample, x='tempo_entrega_dias', y='review_score', 
            trendline="ols", title="Tempo de Entrega vs Avaliação"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        if len(df_filtrado) > 2:
            corr, p_valor = stats.pearsonr(df_filtrado['tempo_entrega_dias'].fillna(0), df_filtrado['review_score'].fillna(0))
            st.info(f"**Coeficiente de Correlação (r):** {corr:.3f} | **P-Valor:** {p_valor:.5f}")
            
        st.divider()
        st.header("Testes de Hipótese (A/B Testing)")
        metrica_teste = st.radio("Métrica a analisar:", ["Tempo de Entrega", "Valor do Frete"])
        coluna_metrica = 'tempo_entrega_dias' if "Tempo" in metrica_teste else 'freight_value'
        
        col_sel1, col_sel2 = st.columns(2)
        estado_a = col_sel1.selectbox("Estado A", estados_disponiveis, index=0)
        estado_b = col_sel2.selectbox("Estado B", estados_disponiveis, index=1 if len(estados_disponiveis)>1 else 0)
        
        dados_a = df[df['customer_state'] == estado_a][coluna_metrica].dropna()
        dados_b = df[df['customer_state'] == estado_b][coluna_metrica].dropna()
        
        if len(dados_a) > 5 and len(dados_b) > 5:
            _, p_valor_t = stats.ttest_ind(dados_a, dados_b, equal_var=False)
            c_t1, c_t2 = st.columns(2)
            c_t1.metric(f"Média {estado_a}", f"{dados_a.mean():.2f}")
            c_t2.metric(f"Média {estado_b}", f"{dados_b.mean():.2f}")
            
            if p_valor_t < 0.05:
                st.success(f"✅ Diferença Estatisticamente Significativa (p-valor: {p_valor_t:.4e})")
            else:
                st.warning(f"❌ Diferença não é estatisticamente significativa (p-valor: {p_valor_t:.4e})")

    # --- ABA 4: SEGMENTAÇÃO RFM ---
    with tab4:
        st.header("Análise RFM de Clientes")
        data_atual = df_filtrado['order_purchase_timestamp'].max()
        rfm = df_filtrado.groupby('customer_unique_id').agg(
            Recencia=('order_purchase_timestamp', lambda x: (data_atual - x.max()).days),
            Frequencia=('order_id', 'nunique'),
            Monetario=('valor_total_item', 'sum')
        ).reset_index()

        def segmentar(row):
            if row['Recencia'] <= 30 and row['Frequencia'] > 1: return '🌟 Campeões'
            elif row['Recencia'] > 180 and row['Frequencia'] == 1: return '⚠️ Risco de Perda'
            elif row['Monetario'] > rfm['Monetario'].quantile(0.75): return '💰 Alto Valor'
            else: return '🚶 Clientes Comuns'

        rfm['Segmento'] = rfm.apply(segmentar, axis=1)
        fig_rfm = px.pie(rfm, names='Segmento', title="Distribuição da Base de Clientes", hole=0.4)
        st.plotly_chart(fig_rfm, use_container_width=True)

    # --- ABA 5: MACHINE LEARNING ---
    with tab5:
        st.header("Motor de Previsão de Entrega")
        modelo, features, r2_score = treinar_modelo_previsao(df)
        
        if modelo:
            st.info(f"🤖 Modelo treinado. Acurácia de tendência (R²): {r2_score:.2f}")
            c_form1, c_form2, c_form3 = st.columns(3)
            sim_frete = c_form1.number_input("Frete (R$):", value=20.0)
            sim_dist = c_form2.number_input("Distância (KM):", value=500.0)
            sim_preco = c_form3.number_input("Preço (R$):", value=100.0)
            
            if st.button("Prever Tempo de Entrega"):
                previsao = modelo.predict([[sim_frete, sim_dist, sim_preco]])
                st.success(f"📦 Tempo Estimado: **{previsao[0]:.0f} dias**")
        else:
            st.warning("Dados insuficientes para treinar o modelo.")
    # --- ABA 6: EXPLORADOR DE DADOS ---
    with tab6:
        st.header("🗄️ Explorador de Dados (Dataset)")
        st.markdown("Visualiza as linhas e colunas que estão a alimentar este painel. Usa os filtros na barra lateral para restringir os dados.")
        
        # Mostra o dataframe interativo no ecrã
        st.dataframe(df_filtrado, use_container_width=True)
        
        # Bónus: Botão para descarregar os dados filtrados
        st.divider()
        st.subheader("Exportar Dados")
        st.markdown("Queres analisar estes dados filtrados no Excel?")
        
        # Converte o dataframe para CSV de forma otimizada
        csv = df_filtrado.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Descarregar Dados Filtrados (CSV)",
            data=csv,
            file_name='dataset_olist_filtrado.csv',
            mime='text/csv',
        )
if __name__ == "__main__":
    main()