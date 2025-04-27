import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split

# Configuración de la página
st.set_page_config(
    page_title="Dashboard Ventas Electrónicos",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Funciones auxiliares
def get_filters(df):
    branches = df['Sucursal'].unique().tolist()
    selected_branches = st.sidebar.multiselect(
        "Seleccionar Sucursales", branches, default=branches
    )
    min_date, max_date = df['Fecha'].min(), df['Fecha'].max()
    date_range = st.sidebar.date_input(
        "Rango de Fechas", [min_date, max_date],
        min_value=min_date, max_value=max_date
    )
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    n_clusters = st.sidebar.slider(
        "Número de Clusters", min_value=2, max_value=6, value=3
    )
    return selected_branches, start_date, end_date, n_clusters


def show_metrics(df):
    total_sales = df['Venta_Total'].sum()
    total_tx = df.shape[0]
    avg_ticket = total_sales / total_tx if total_tx else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Ventas Totales ($)", f"{total_sales:,.2f}")
    c2.metric("Total Transacciones", total_tx)
    c3.metric("Ticket Promedio ($)", f"{avg_ticket:,.2f}")


def clustering(df, n_clusters):
    st.subheader("🔵 Clusterización de Sucursales por Producto")
    # Pivot por sucursal en lugar de Cliente_Tipo para obtener más puntos
    pivot = df.pivot_table(
        index='Sucursal',
        columns='Producto',
        values='Cantidad',
        aggfunc='sum',
        fill_value=0
    )
    X = pivot.values
    k = min(n_clusters, X.shape[0])
    kmeans = KMeans(n_clusters=k, random_state=42)
    pivot['Cluster'] = kmeans.fit_predict(X)
    fig = px.scatter(
        pivot.reset_index(),
        x=pivot.columns[0],
        y=pivot.columns[1],
        color='Cluster',
        hover_data=['Sucursal'],
        title="Clusters de Sucursales según mix de productos"
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)
    st.write("Agrupación de sucursales por similitud en ventas de productos.")


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
    fig = px.bar(
        dfp,
        x='Sucursal',
        y='Predicción',
        title="Predicción Ventas Próximo Mes por Sucursal"
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)
    st.write("Estimación de ventas para el siguiente mes.")


def anomalies(df):
    st.subheader("🟥 Detección de Anomalías en Ventas")
    isol = IsolationForest(contamination=0.05, random_state=42)
    df['Anomalia'] = isol.fit_predict(df[['Precio', 'Cantidad']])
    fig = px.scatter(
        df,
        x='Fecha',
        y='Venta_Total',
        color=df['Anomalia'].map({1: 'Normal', -1: 'Anómalo'}),
        title="Anomalías en Ventas por Fecha"
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)
    st.write("Ventas que se salen del patrón normal.")


def correlation(df):
    st.subheader("🔴 Mapa de Correlación")
    corr = df[['Precio', 'Cantidad', 'Venta_Total']].corr()
    fig = px.imshow(corr, text_auto=True, title="Correlación de Variables")
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)


def insights(df):
    st.subheader("📊 Otros Insights")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("🟡 Top 5 Productos")
        top = df.groupby('Producto')['Cantidad'].sum().nlargest(5).reset_index()
        fig_top = px.bar(top, x='Producto', y='Cantidad', title="Top 5 Productos")
        fig_top.update_layout(height=350)
        st.plotly_chart(fig_top, use_container_width=True)
    with col2:
        st.write("🟠 Métodos de Pago")
        pay = df['Método_Pago'].value_counts().reset_index()
        pay.columns = ['Metodo', 'Cantidad']
        fig_pay = px.pie(pay, names='Metodo', values='Cantidad', title='Métodos de Pago')
        fig_pay.update_layout(height=350)
        st.plotly_chart(fig_pay, use_container_width=True)
    with col3:
        st.write("🟢 Tendencia Mensual")
        trend = df.groupby('Mes')['Venta_Total'].sum().reset_index()
        fig_trend = px.line(trend, x='Mes', y='Venta_Total', title='Tendencia Mensual')
        fig_trend.update_layout(height=350)
        st.plotly_chart(fig_trend, use_container_width=True)


# Contenido principal
st.sidebar.title("🔧 Filtros y Configuración")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df['Venta_Total'] = df['Precio'] * df['Cantidad']
    df['Mes'] = df['Fecha'].dt.to_period('M').astype(str)

    # Aplicar filtros
    selected_branches, start_date, end_date, n_clusters = get_filters(df)
    df = df[df['Sucursal'].isin(selected_branches)]
    df = df[(df['Fecha'] >= start_date) & (df['Fecha'] <= end_date)]

    # Encabezado
    st.title("📈 Dashboard Ventas Electrónicos")
    st.write(f"Registros mostrados: {len(df)}")
    st.markdown("---")

    # Mostrar métricas y gráficas
    show_metrics(df)
    clustering(df, n_clusters)
    prediction(df)
    anomalies(df)
    correlation(df)
    insights(df)

else:
    st.sidebar.info("Por favor sube un archivo CSV para comenzar.")
