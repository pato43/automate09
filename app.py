import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split

# Configuración de página
st.set_page_config(page_title="Dashboard Ventas Electrónicos", layout="wide")

st.title("📈 Dashboard de Ventas de Electrónicos")
st.write("Sube tu archivo CSV para analizar las ventas de forma interactiva.")

# Subir archivo CSV
uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Preparar datos
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df['Venta_Total'] = df['Precio'] * df['Cantidad']
    df['Mes'] = df['Fecha'].dt.to_period('M').astype(str)

    st.subheader("Vista previa de los datos")
    st.dataframe(df.head())
    st.markdown("---")

    # Métricas principales
    total_ventas = df['Venta_Total'].sum()
    total_transacciones = df.shape[0]
    ticket_promedio = total_ventas / total_transacciones

    m1, m2, m3 = st.columns(3)
    m1.metric("Ventas Totales ($)", f"{total_ventas:,.2f}")
    m2.metric("Total de Transacciones", total_transacciones)
    m3.metric("Ticket Promedio ($)", f"{ticket_promedio:,.2f}")

    st.markdown("---")

    # 1. Clusterización de Clientes por Producto
    st.subheader("🔵 Clusterización de Clientes por Producto")
    cluster_data = df.pivot_table(
        index='Cliente_Tipo', columns='Producto', values='Cantidad', aggfunc='sum', fill_value=0
    )
    Xc = cluster_data.values
    n_clusters = min(3, Xc.shape[0])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(Xc)
    cluster_data['Cluster'] = labels
    fig_cluster = px.scatter(
        cluster_data.reset_index(), x=cluster_data.columns[0], y=cluster_data.columns[1],
        color='Cluster', title="Clusters de Clientes según Productos Comprados"
    )
    fig_cluster.update_layout(height=450, margin=dict(l=40, r=40, t=60, b=40))
    st.write("Este gráfico muestra cómo se agrupan los clientes nuevos y recurrentes según su patrón de compra de productos.")

    # 2. Predicción de Ventas por Sucursal (Próximo Mes)
    st.subheader("🟣 Predicción de Ventas por Sucursal (Próximo Mes)")
    df['MesNum'] = df['Fecha'].dt.year * 12 + df['Fecha'].dt.month
    ventas_mes_suc = df.groupby(['Sucursal', 'MesNum'])['Venta_Total'].sum().reset_index()
    X_train = pd.get_dummies(ventas_mes_suc['Sucursal'])
    X_train['MesNum'] = ventas_mes_suc['MesNum']
    y_train = ventas_mes_suc['Venta_Total']
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
    last_mes = ventas_mes_suc['MesNum'].max()
    next_mes = last_mes + 1
    sucursales = ventas_mes_suc['Sucursal'].unique()
    df_pred = pd.DataFrame({'Sucursal': sucursales, 'MesNum': next_mes})
    X_pred = pd.get_dummies(df_pred.set_index('Sucursal')).reset_index()
    X_pred = X_pred.reindex(columns=X_train.columns, fill_value=0)
    df_pred['Predicción'] = rf.predict(X_pred)
    fig_pred = px.bar(
        df_pred, x='Sucursal', y='Predicción',
        title="Predicción de Ventas Próximo Mes por Sucursal"
    )
    fig_pred.update_layout(height=400, margin=dict(l=40, r=40, t=60, b=40))
    st.write("Usando RandomForest, estimamos las ventas del mes siguiente para cada sucursal.")

    # 3. Detección de Anomalías
    st.subheader("🟥 Detección de Anomalías en Precios y Cantidades")
    iso = IsolationForest(contamination=0.05, random_state=42)
    df['Anomalia'] = iso.fit_predict(df[['Precio', 'Cantidad']])
    fig_anom = px.scatter(
        df, x='Precio', y='Cantidad',
        color=df['Anomalia'].map({1: 'Normal', -1: 'Anómalo'}),
        title="Anomalías detectadas en ventas"
    )
    fig_anom.update_layout(height=400, margin=dict(l=40, r=40, t=60, b=40))
    st.write("Se identifican ventas con comportamiento atípico en precio o cantidad.")

    # Mostrar gráficos principales con tamaños proporcionales
    st.markdown("### 🔍 Análisis General")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.plotly_chart(fig_cluster, use_container_width=True)
    with col2:
        st.plotly_chart(fig_pred, use_container_width=True)
    with col3:
        st.plotly_chart(fig_anom, use_container_width=True)

    st.markdown("---")

    # 4. Productos más vendidos por zona
    st.subheader("🟡 Productos más vendidos por zona")
    top_prod = df.groupby(['Sucursal', 'Producto'])['Cantidad'].sum().reset_index()
    fig_top = px.bar(
        top_prod, x='Producto', y='Cantidad', color='Sucursal', barmode='group',
        title="Top Productos por Sucursal"
    )
    fig_top.update_layout(height=350, margin=dict(l=40, r=40, t=60, b=40))
    st.write("Ranking de productos con mayor volumen de ventas en cada sucursal.")

    # 5. Distribución de Métodos de Pago
    st.subheader("🟠 Distribución de Métodos de Pago")
    pay = df['Método_Pago'].value_counts().reset_index()
    pay.columns = ['Metodo', 'Cantidad']
    fig_pay = px.pie(
        pay, names='Metodo', values='Cantidad', title='Métodos de Pago Utilizados'
    )
    fig_pay.update_layout(height=350, margin=dict(l=40, r=40, t=60, b=40))
    st.write("Porcentaje de uso de cada método de pago.")

    # 6. Evolución Mensual de Ventas
    st.subheader("🟢 Evolución Mensual de Ventas")
    ventas_mes = df.groupby('Mes')['Venta_Total'].sum().reset_index()
    fig_mes = px.line(
        ventas_mes, x='Mes', y='Venta_Total', title='Tendencia de Ventas Mensuales'
    )
    fig_mes.update_layout(height=350, margin=dict(l=40, r=40, t=60, b=40))
    st.write("Tendencia de ventas a lo largo del año.")

    # Segunda fila de gráficos secundarios
    st.markdown("### 📊 Otros Insights")
    sec1, sec2, sec3 = st.columns([1, 1, 1])
    with sec1:
        st.plotly_chart(fig_top, use_container_width=True)
    with sec2:
        st.plotly_chart(fig_pay, use_container_width=True)
    with sec3:
        st.plotly_chart(fig_mes, use_container_width=True)

else:
    st.info("Por favor sube un archivo CSV para comenzar.")
