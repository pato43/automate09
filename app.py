import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split

# Configuración de la página
st.set_page_config(
    page_title="Dashboard Ventas Electrónicos",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar para filtros y configuración
st.sidebar.title("🔧 Filtros y Configuración")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv"])

if uploaded_file:
    # Cargar y preparar datos
    df = pd.read_csv(uploaded_file)
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df['Venta_Total'] = df['Precio'] * df['Cantidad']
    df['Mes'] = df['Fecha'].dt.to_period('M').astype(str)

    # Filtro por sucursal
def get_filters(df):
    branches = df['Sucursal'].unique().tolist()
    selected_branches = st.sidebar.multiselect("Seleccionar Sucursales", branches, default=branches)
    # Filtro por rango de fechas
    min_date, max_date = df['Fecha'].min(), df['Fecha'].max()
    date_range = st.sidebar.date_input(
        "Rango de Fechas", [min_date, max_date], min_value=min_date, max_value=max_date
    )
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    # Número de clusters
    n_clusters = st.sidebar.slider("Número de Clusters", min_value=2, max_value=6, value=3)
    return selected_branches, start_date, end_date, n_clusters

    selected_branches, start_date, end_date, n_clusters = get_filters(df)
    df = df[df['Sucursal'].isin(selected_branches)]
    df = df[(df['Fecha'] >= start_date) & (df['Fecha'] <= end_date)]

    # Título y descripción
    st.title("📈 Dashboard de Ventas de Electrónicos")
    st.write(f"Visualización interactiva: {len(df)} registros filtrados.")
    st.markdown("---")

    # Métricas principales
def show_metrics(df):
    total_sales = df['Venta_Total'].sum()
    total_tx = df.shape[0]
    avg_ticket = total_sales / total_tx if total_tx else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Ventas Totales ($)", f"{total_sales:,.2f}")
    c2.metric("Total Transacciones", total_tx)
    c3.metric("Ticket Promedio ($)", f"{avg_ticket:,.2f}")

    show_metrics(df)
    st.markdown("---")

    # 1. Clusterización de Clientes por Producto
def clustering(df, n_clusters):
    st.subheader("🔵 Clusterización de Clientes por Producto")
    pivot = df.pivot_table(index='Cliente_Tipo', columns='Producto', values='Cantidad', aggfunc='sum', fill_value=0)
    X = pivot.values
    kmeans = KMeans(n_clusters=min(n_clusters, X.shape[0]), random_state=42)
    pivot['Cluster'] = kmeans.fit_predict(X)
    fig = px.scatter(
        pivot.reset_index(), x=pivot.columns[0], y=pivot.columns[1], color='Cluster',
        title="Clusters de Clientes según Productos Comprados"
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)
    st.write("Agrupación de clientes según patrones de compra.")

    clustering(df, n_clusters)

    # 2. Predicción de Ventas por Sucursal (Próximo Mes)
def prediction(df):
    st.subheader("🟣 Predicción de Ventas por Sucursal (Próximo Mes)")
    df['MesNum'] = df['Fecha'].dt.year * 12 + df['Fecha'].dt.month
    grouped = df.groupby(['Sucursal', 'MesNum'])['Venta_Total'].sum().reset_index()
    X = pd.get_dummies(grouped['Sucursal'])
    X['MesNum'] = grouped['MesNum']
    y = grouped['Venta_Total']
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    last_mes = grouped['MesNum'].max()
    next_mes = last_mes + 1
    subs = grouped['Sucursal'].unique()
    dfp = pd.DataFrame({'Sucursal': subs, 'MesNum': next_mes})
    Xp = pd.get_dummies(dfp.set_index('Sucursal')).reset_index()
    Xp = Xp.reindex(columns=X.columns, fill_value=0)
    dfp['Predicción'] = model.predict(Xp)
    fig = px.bar(dfp, x='Sucursal', y='Predicción', title="Predicción Ventas Próximo Mes")
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)
    st.write("Estimación de ventas para el siguiente mes.")

    prediction(df)

    # 3. Detección de Anomalías
def anomalies(df):
    st.subheader("🟥 Detección de Anomalías en Ventas")
    isol = IsolationForest(contamination=0.05, random_state=42)
    df['Anomalia'] = isol.fit_predict(df[['Precio', 'Cantidad']])
    fig = px.scatter(
        df, x='Fecha', y='Venta_Total', color=df['Anomalia'].map({1:'Normal',-1:'Anómalo'}),
        title="Anomalías en Ventas por Fecha"
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)
    st.write("Ventas que se salen del patrón normal.")

    anomalies(df)
    st.markdown("---")

    # 4. Correlación de Variables
    st.subheader("🔴 Mapa de Correlación")
    corr = df[['Precio','Cantidad','Venta_Total']].corr()
    fig_corr = px.imshow(corr, text_auto=True, title="Correlación de Variables")
    fig_corr.update_layout(height=350)
    st.plotly_chart(fig_corr, use_container_width=True)

    # 5. Productos top, pagos y tendencia
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("🟡 Top Productos")
        top = df.groupby('Producto')['Cantidad'].sum().nlargest(5).reset_index()
        st.bar_chart(top.set_index('Producto'))
    with col2:
        st.subheader("🟠 Métodos de Pago")
        pay = df['Método_Pago'].value_counts()
        st.pie_chart(pay)
    with col3:
        st.subheader("🟢 Tendencia Mensual")
        trend = df.groupby('Mes')['Venta_Total'].sum()
        st.line_chart(trend)

else:
    st.sidebar.info("Por favor sube un archivo CSV para comenzar.")
