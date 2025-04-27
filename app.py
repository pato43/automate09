import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split

# Configurar página
st.set_page_config(page_title="Dashboard Ventas Electrónicos", layout="wide")

st.title("📈 Dashboard de Ventas de Electrónicos")
st.write("Sube tu archivo CSV para analizar las ventas.")

# Subida de archivo
uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Vista previa de los datos:")
    st.dataframe(df.head())

    st.markdown("---")

    # 1. Estadísticas generales
    st.subheader("Resumen General")
    col1, col2, col3 = st.columns(3)

    with col1:
        total_ventas = (df['Precio'] * df['Cantidad']).sum()
        st.metric("Ventas Totales ($)", f"{total_ventas:,.2f}")

    with col2:
        total_transacciones = df.shape[0]
        st.metric("Total de Ventas", total_transacciones)

    with col3:
        ticket_promedio = total_ventas / total_transacciones
        st.metric("Ticket Promedio ($)", f"{ticket_promedio:,.2f}")

    st.markdown("---")

    # 2. Clusterización de clientes basado en productos
    st.subheader("Agrupación de Clientes por Producto")
    # CORREGIDO: usar pivot_table para no crear un multi-index que causa error en KMeans
    cluster_data = df.pivot_table(index='Cliente_Tipo', columns='Producto', values='Cantidad', aggfunc='sum', fill_value=0)
    X = cluster_data.values
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    cluster_data['Cluster'] = cluster_labels

    fig_cluster = px.scatter(cluster_data.reset_index(), 
                              x=cluster_data.columns[0], 
                              y=cluster_data.columns[1], 
                              color='Cluster',
                              title="Clusters de Clientes por Productos Comprados")
    st.plotly_chart(fig_cluster, use_container_width=True)

    st.markdown("---")

    # 3. Predicción de ventas por sucursal
    st.subheader("Predicción de Ventas por Sucursal")
    df['Venta_Total'] = df['Precio'] * df['Cantidad']
    ventas_sucursal = df.groupby('Sucursal')['Venta_Total'].sum().reset_index()

    X_train, X_test, y_train, y_test = train_test_split(
        pd.get_dummies(ventas_sucursal['Sucursal']), ventas_sucursal['Venta_Total'], test_size=0.2, random_state=42
    )

    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    predicciones = model.predict(X_test)

    fig_pred = px.bar(ventas_sucursal, x='Sucursal', y='Venta_Total', title="Ventas Reales por Sucursal")
    st.plotly_chart(fig_pred, use_container_width=True)

    st.markdown("---")

    # 4. Detección de anomalías
    st.subheader("Detección de Anomalías (Precios y Cantidades)")
    anomaly_detector = IsolationForest(contamination=0.05)
    df['Anomalia'] = anomaly_detector.fit_predict(df[['Precio', 'Cantidad']])

    fig_anomalias = px.scatter(df, x='Precio', y='Cantidad', 
                               color=df['Anomalia'].map({1: 'Normal', -1: 'Anómalo'}),
                               title="Anomalías detectadas en Precios y Cantidades")
    st.plotly_chart(fig_anomalias, use_container_width=True)

    st.markdown("---")

    # 5. Productos más vendidos por zona
    st.subheader("Productos Más Vendidos por Zona")
    top_productos = df.groupby(['Sucursal', 'Producto'])['Cantidad'].sum().reset_index()
    fig_top = px.bar(top_productos, x='Producto', y='Cantidad', color='Sucursal', barmode='group',
                     title="Productos más vendidos por sucursal")
    st.plotly_chart(fig_top, use_container_width=True)

    st.markdown("---")

    # 6. Método de pago
    st.subheader("Distribución de Métodos de Pago")
    metodo_pago = df['Método_Pago'].value_counts().reset_index()
    metodo_pago.columns = ['Metodo', 'Cantidad']
    fig_pago = px.pie(metodo_pago, names='Metodo', values='Cantidad', title='Uso de Método de Pago')
    st.plotly_chart(fig_pago, use_container_width=True)

    st.markdown("---")

    # 7. Evolución mensual de ventas
    st.subheader("Evolución Mensual de Ventas")
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df['Mes'] = df['Fecha'].dt.to_period('M')
    ventas_mes = df.groupby('Mes')['Venta_Total'].sum().reset_index()
    fig_mes = px.line(ventas_mes, x='Mes', y='Venta_Total', title='Ventas Acumuladas por Mes')
    st.plotly_chart(fig_mes, use_container_width=True)

else:
    st.info("Por favor sube un archivo CSV para comenzar.")
