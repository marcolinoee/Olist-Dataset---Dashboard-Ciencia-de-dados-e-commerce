import pandas as pd
import numpy as np

def extract() -> dict:
    """Carrega os datasets necessários em um dicionário."""
    print("Extraindo dados...")
    arquivos = {
        'orders': 'olist_orders_dataset.csv',
        'items': 'olist_order_items_dataset.csv',
        'customers': 'olist_customers_dataset.csv',
        'reviews': 'olist_order_reviews_dataset.csv'
    }
    return {nome: pd.read_csv(caminho) for nome, caminho in arquivos.items()}

def transform(dfs: dict) -> pd.DataFrame:
    """Cruza as tabelas e cria variáveis estatísticas."""
    print("Transformando dados...")
    
    # Joins
    df = dfs['orders'].merge(dfs['customers'], on='customer_id', how='inner')
    df = df.merge(dfs['items'], on='order_id', how='inner')
    
    # Pegar apenas a avaliação mais recente de cada pedido para evitar duplicatas
    reviews_unicos = dfs['reviews'].drop_duplicates(subset=['order_id'], keep='last')
    df = df.merge(reviews_unicos[['order_id', 'review_score']], on='order_id', how='left')
    
    # Tratamento de Datas
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
    
    # Engenharia de Features (Variáveis para Estatística)
    df['valor_total'] = df['price'] + df['freight_value']
    
    # Calcular Tempo de Entrega em dias
    df['tempo_entrega_dias'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
    
    # Limpeza
    df_clean = df.dropna(subset=['tempo_entrega_dias', 'review_score', 'valor_total']).copy()
    
    # Remover Outliers absurdos (ex: entregas que demoraram mais de 100 dias)
    df_clean = df_clean[df_clean['tempo_entrega_dias'] < 100]
    
    return df_clean

def load(df: pd.DataFrame, filename: str):
    """Salva o dataset analítico final."""
    print(f"Carregando dados para {filename}...")
    # Salvando em Parquet para maior performance
    df.to_parquet(filename, index=False)
    print("ETL Concluído com Sucesso!")

if __name__ == "__main__":
    datasets = extract()
    df_final = transform(datasets)
    load(df_final, 'dados_estatisticos.parquet')