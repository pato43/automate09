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
st.write("Sube tu archivo CSV para analizar las ventas.")

# Cargar archivo
uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Vista previa de los datos")
    st.dataframe(df.head())
    st.markdown("---")

    # Métricas generales
    total_ventas = (df['Precio'] * df['Cantidad']).sum()
    total_transacciones = df.shape[0]
    ticket_promedio = total_ventas / total_transacciones

    cols = st.columns(3)
    cols[0].metric("Ventas Totales ($)", f"{total_ventas:,.2f}")
    cols[1].metric("Total de Ventas", total_transacciones)
    cols[2].metric("Ticket Promedio ($)", f"{ticket_promedio:,.2f}")

    st.markdown("---")

    # 1. Clusterización de Clientes por Producto
    st.subheader("🔵 Clusterización de Clientes por Producto")
    cluster_data = df.pivot_table(
        index='Cliente_Tipo',
        columns='Producto',
        values='Cantidad',
        aggfunc='sum',
        fill_value=0
    )
    X = cluster_data.values
    n_samples = X.shape[0]
    n_clusters = min(3, n_samples)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    cluster_data['Cluster'] = labels
    fig_cluster = px.scatter(
        cluster_data.reset_index(),
        x=cluster_data.columns[0],
        y=cluster_data.columns[1],
        color='Cluster',
        title="Clientes agrupados por productos"
    )

    # 2. Predicción de Ventas por Sucursal
    st.subheader("🟣 Predicción de Ventas por Sucursal")
    df['Venta_Total'] = df['Precio'] * df['Cantidad']
    ventas_s = df.groupby('Sucursal')['Venta_Total'].sum().reset_index()
    X_train, X_test, y_train, y_test = train_test_split(
        pd.get_dummies(ventas_s['Sucursal']),
        ventas_s['Venta_Total'],
        test_size=0.2,
        random_state=42
    )
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    fig_pred = px.bar(
        ventas_s,
        x='Sucursal',
        y='Venta_Total',
        title="Ventas Reales por Sucursal"
    )

    # 3. Detección de Anomalías
    st.subheader("🟥 Detección de Anomalías en Precios y Cantidades")
    iso = IsolationForest(contamination=0.05, random_state=42)
    df['Anomalia'] = iso.fit_predict(df[['Precio', 'Cantidad']])
    fig_anom = px.scatter(
        df,
        x='Precio',
        y='Cantidad',
        color=df['Anomalia'].map({1: 'Normal', -1: 'Anómalo'}),
        title="Detección de Anomalías"
    )

    # Mostrar las tres gráficas principales en una fila
    st.markdown("### 🔍 Análisis General")
    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(fig_cluster, use_container_width=True)
    with g2:
        st.plotly_chart(fig_pred, use_container_width=True)
    with g3:
        st.plotly_chart(fig_anom, use_container_width=True)

    st.markdown("---")

    # 4. Productos más vendidos por zona
    st.subheader("🟡 Productos más vendidos por Zona")
    top_prod = df.groupby(['Sucursal', 'Producto'])['Cantidad'].sum().reset_index()
    fig_top = px.bar(
        top_prod,
        x='Producto',
        y='Cantidad',
        color='Sucursal',
        barmode='group',
        title="Top Productos por Sucursal"
    )

    # 5. Distribución de Métodos de Pago
    st.subheader("🟠 Distribución de Métodos de Pago")
    pay = df['Método_Pago'].value_counts().reset_index()
    pay.columns = ['Metodo', 'Cantidad']
    fig_pay = px.pie(
        pay,
        names='Metodo',
        values='Cantidad',
        title='Métodos de Pago'
    )

    # 6. Evolución Mensual de Ventas
    st.subheader("🟢 Evolución Mensual de Ventas")
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df['Mes'] = df['Fecha'].dt.to_period('M').astype(str)
    ventas_mes = df.groupby('Mes')['Venta_Total'].sum().reset_index()
    fig_mes = px.line(
        ventas_mes,
        x='Mes',
        y='Venta_Total',
        title='Ventas por Mes'
    )

    # Mostrar los insights secundarios en fila
    st.markdown("### 📊 Otros Insights")
    h1, h2, h3 = st.columns(3)
    with h1:
        st.plotly_chart(fig_top, use_container_width=True)
    with h2:
        st.plotly_chart(fig_pay, use_container_width=True)
    with h3:
        st.plotly_chart(fig_mes, use_container_width=True)

else:
    st.info("Por favor sube un archivo CSV para comenzar.")
