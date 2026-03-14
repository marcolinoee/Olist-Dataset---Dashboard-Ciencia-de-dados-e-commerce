import pandas as pd
import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    """
    Calcula a distância em KM entre dois pontos (Latitude e Longitude) na Terra.
    """
    # Converter graus para radianos
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Fórmula de Haversine
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # Raio da Terra em KM
    return c * r

def extract() -> dict:
    """Carrega TODOS os datasets disponíveis no ecossistema Olist."""
    print("Extraindo dados de todas as fontes...")
    arquivos = {
        'orders': 'olist_orders_dataset.csv',
        'items': 'olist_order_items_dataset.csv',
        'customers': 'olist_customers_dataset.csv',
        'reviews': 'olist_order_reviews_dataset.csv',
        'sellers': 'olist_sellers_dataset.csv',
        'payments': 'olist_order_payments_dataset.csv',
        'products': 'olist_products_dataset.csv',
        'translation': 'product_category_name_translation.csv',
        'geo': 'olist_geolocation_dataset.csv'
    }
    return {nome: pd.read_csv(caminho) for nome, caminho in arquivos.items()}

def transform(dfs: dict) -> pd.DataFrame:
    """Cruza as 9 tabelas e cria a Tabela Analítica de Dados (ABT)."""
    print("Transformando e cruzando dados...")
    
    # 1. Base principal: Pedidos + Clientes
    df = dfs['orders'].merge(dfs['customers'], on='customer_id', how='inner')
    
    # 2. + Itens (Granularidade: 1 linha por item do pedido)
    df = df.merge(dfs['items'], on='order_id', how='inner')
    
    # 3. + Produtos e Tradução
    df = df.merge(dfs['products'], on='product_id', how='left')
    df = df.merge(dfs['translation'], on='product_category_name', how='left')
    
    # 4. + Vendedores
    df = df.merge(dfs['sellers'], on='seller_id', how='left')
    
    # 5. + Avaliações (Tratando duplicatas: pegar a última avaliação do pedido)
    reviews_unicos = dfs['reviews'].sort_values('review_creation_date').drop_duplicates(subset=['order_id'], keep='last')
    df = df.merge(reviews_unicos[['order_id', 'review_score']], on='order_id', how='left')
    
    # 6. + Pagamentos (Agrupando pois um pedido pode ter múltiplos pagamentos/cartões)
    pagamentos_agrupados = dfs['payments'].groupby('order_id').agg(
        total_pago=('payment_value', 'sum'),
        max_parcelas=('payment_installments', 'max'),
        tipo_pagamento_principal=('payment_type', lambda x: x.mode()[0] if not x.mode().empty else 'unknown')
    ).reset_index()
    df = df.merge(pagamentos_agrupados, on='order_id', how='left')
    
    # 7. + Geolocalização (Pegando 1 Lat/Lng por CEP para não explodir a base)
    geo_unica = dfs['geo'].drop_duplicates(subset=['geolocation_zip_code_prefix']).copy()
    geo_unica.rename(columns={
        'geolocation_zip_code_prefix': 'zip_code',
        'geolocation_lat': 'lat',
        'geolocation_lng': 'lng'
    }, inplace=True)
    
    # Pegar Lat/Lng do Cliente
    df = df.merge(geo_unica, left_on='customer_zip_code_prefix', right_on='zip_code', how='left')
    df.rename(columns={'lat': 'cliente_lat', 'lng': 'cliente_lng'}, inplace=True)
    
    # Pegar Lat/Lng do Vendedor
    df = df.merge(geo_unica, left_on='seller_zip_code_prefix', right_on='zip_code', how='left')
    df.rename(columns={'lat': 'vendedor_lat', 'lng': 'vendedor_lng'}, inplace=True)
    
    print("Gerando novas variáveis estatísticas...")
    # Engenharia de Features
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
    
    # A. Valor do Carrinho (Preço + Frete)
    df['valor_total_item'] = df['price'] + df['freight_value']
    
    # B. Tempo de Entrega (em dias)
    df['tempo_entrega_dias'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
    
    # C. Distância Logística (KM entre Cliente e Vendedor)
    df['distancia_logistica_km'] = haversine(
        df['cliente_lat'], df['cliente_lng'], 
        df['vendedor_lat'], df['vendedor_lng']
    )
    
    # D. Limpeza Final (Filtro de pedidos entregues e remoção de outliers)
    df_clean = df[df['order_status'] == 'delivered'].copy()
    df_clean = df_clean.dropna(subset=['tempo_entrega_dias', 'review_score', 'valor_total_item', 'distancia_logistica_km'])
    df_clean = df_clean[df_clean['tempo_entrega_dias'] < 100] # Limpar erros sistêmicos de datas
    
    return df_clean

def load(df: pd.DataFrame, filename: str):
    """Salva a base final otimizada."""
    print(f"Base de dados unificada possui {df.shape[0]} linhas e {df.shape[1]} colunas.")
    print(f"Salvando dados otimizados em {filename}...")
    df.to_parquet(filename, index=False)
    print("ETL Concluído com Sucesso Master! 🚀")

if __name__ == "__main__":
    datasets = extract()
    df_final = transform(datasets)
    load(df_final, 'dados_estatisticos.parquet')