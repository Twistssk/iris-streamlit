import io
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Explorador Iris", page_icon="🌸", layout="wide")

COLS = ['largo_sepalo', 'ancho_sepalo', 'largo_petalo', 'ancho_petalo', 'especies']
NUM = COLS[:4]
ETIQ = {'largo_sepalo': 'Largo del sépalo', 'ancho_sepalo': 'Ancho del sépalo',
        'largo_petalo': 'Largo del pétalo', 'ancho_petalo': 'Ancho del pétalo',
        'area_petalo': 'Área del pétalo'}
COLOR = {'Setosa': '#E4572E', 'Versicolor': '#29A19C', 'Virginica': '#3B6FB6'}
fmt = lambda c: ETIQ[c]


@st.cache_data
def cargar(origen):
    """Lee el CSV (sin encabezado) y calcula 'area_petalo'."""
    df = pd.read_csv(origen, header=None, names=COLS)
    df['especies'] = df['especies'].str.strip()
    df['area_petalo'] = df['largo_petalo'] * df['ancho_petalo']
    return df


def fuerza(r):
    a = abs(r)
    nivel = "FUERTE" if a >= 0.7 else "MODERADA" if a >= 0.4 else "DÉBIL"
    return f"{nivel} {'positiva' if r > 0 else 'negativa'}"


def mostrar(fig):
    st.pyplot(fig)
    plt.close(fig)


# ---------------- Origen de datos ----------------
st.sidebar.title("🌸 Panel de control")
archivo = st.sidebar.file_uploader("Cargar otro iris.csv (opcional)", type="csv")
ruta = Path(__file__).parent / "iris.csv"
if archivo is not None:
    df = cargar(archivo)
elif ruta.exists():
    df = cargar(str(ruta))
else:
    st.error("No se encontró 'iris.csv'. Colócalo junto a este script o súbelo en la barra lateral.")
    st.stop()

# ---------------- Filtros globales ----------------
st.sidebar.header("Filtros globales")
especies_sel = st.sidebar.multiselect("Especies", sorted(df['especies'].unique()),
                                      default=sorted(df['especies'].unique()))
mask = df['especies'].isin(especies_sel)
for c in NUM:
    lo, hi = float(df[c].min()), float(df[c].max())
    a, b = st.sidebar.slider(fmt(c) + " (cm)", lo, hi, (lo, hi), step=0.1)
    mask &= df[c].between(a, b)
d = df[mask]
st.sidebar.caption(f"Mostrando {len(d)} de {len(df)} flores")
if d.empty:
    st.warning("Los filtros no dejan ninguna flor. Amplía los rangos en la barra lateral.")
    st.stop()

# ---------------- Encabezado ----------------
st.title("🌸 Explorador Interactivo del Dataset Iris")
st.markdown("Tarea *Flores Iris en Pandas* publicada con Streamlit. "
            "**Todos los resultados responden a los filtros de la barra lateral.**")
c1, c2, c3 = st.columns(3)
c1.metric("Flores", len(d))
c2.metric("Especies", d['especies'].nunique())
c3.metric("Área media del pétalo", f"{d['area_petalo'].mean():.2f} cm²")

tab1, tab2, tab3, tab4 = st.tabs(["📋 1. Exploración", "🧮 2. Análisis",
                                  "📈 3. Visualización", "💡 4. Reflexión"])

# ---------------- Parte 1 ----------------
with tab1:
    st.subheader("1.1 Primeras filas")
    n = st.slider("Cantidad de filas a mostrar", 5, 50, 10)
    st.dataframe(d.head(n))

    st.subheader("1.2 Información general")
    col_a, col_b = st.columns(2)
    buf = io.StringIO()
    d.info(buf=buf)
    col_a.text(buf.getvalue())
    col_b.markdown(f"**Dimensiones:** {d.shape[0]} filas × {d.shape[1]} columnas")
    col_b.markdown("**Valores nulos por columna:**")
    col_b.dataframe(d.isnull().sum().to_frame("Nulos"))

    st.subheader("1.3 Estadísticas descriptivas")
    st.dataframe(d.describe())

    st.subheader("1.4 Cantidad de flores por especie")
    cuenta = d['especies'].value_counts()
    col_a, col_b = st.columns([1, 2])
    col_a.dataframe(cuenta.to_frame("Frecuencia"))
    col_b.bar_chart(cuenta)
    st.download_button("⬇️ Descargar datos filtrados (CSV)", d.to_csv(index=False),
                       "iris_filtrado.csv", "text/csv")

# ---------------- Parte 2 ----------------
with tab2:
    st.subheader("2.1 Media por especie")
    var = st.selectbox("Variable", NUM, index=2, format_func=fmt, key="v21")
    st.dataframe(d.groupby('especies')[var].mean().to_frame(f"Media de {fmt(var).lower()} (cm)"))

    st.subheader("2.2 Flor con el sépalo más ancho")
    st.dataframe(d[d['ancho_sepalo'] == d['ancho_sepalo'].max()])

    st.subheader("2.3 Correlación entre dos variables")
    cx, cy = st.columns(2)
    vx = cx.selectbox("Variable X", NUM, index=2, format_func=fmt, key="x23")
    vy = cy.selectbox("Variable Y", NUM, index=3, format_func=fmt, key="y23")
    if len(d) > 1 and d[vx].std() > 0 and d[vy].std() > 0:
        r = np.corrcoef(d[vx], d[vy])[0, 1]
        st.metric("Coeficiente de correlación (NumPy)", f"{r:.4f}")
        st.info(f"Correlación {fuerza(r)}. Cercano a ±1 indica relación lineal fuerte.")
    else:
        st.info("Se necesitan al menos 2 flores con valores distintos para calcular la correlación.")

    st.subheader("2.4 Columna 'area_petalo' (largo × ancho)")
    st.dataframe(d[['especies', 'largo_petalo', 'ancho_petalo', 'area_petalo']].head(10))

    st.subheader("2.5 Filtro personalizado por especie y largo de sépalo")
    cs, cu = st.columns(2)
    esp = cs.selectbox("Especie", sorted(df['especies'].unique()), key="e25")
    umbral = cu.slider("Largo de sépalo mayor a (cm)", 4.0, 8.0, 5.0, 0.1)
    res = d[(d['especies'] == esp) & (d['largo_sepalo'] > umbral)]
    st.dataframe(res)
    st.markdown(f"**Total encontrado:** {len(res)} flores")

# ---------------- Parte 3 ----------------
with tab3:
    st.subheader("3.1 Histograma")
    h1, h2, h3 = st.columns(3)
    vh = h1.selectbox("Variable", NUM, format_func=fmt, key="h")
    bins = h2.slider("Número de bins", 5, 40, 20)
    sep = h3.checkbox("Separar por especie", value=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if sep:
        for e, g in d.groupby('especies'):
            ax.hist(g[vh], bins=bins, alpha=0.6, label=e, color=COLOR[e], edgecolor='black')
        ax.legend()
    else:
        ax.hist(d[vh], bins=bins, color='skyblue', edgecolor='black')
    ax.set(xlabel=f"{fmt(vh)} (cm)", ylabel="Frecuencia", title=f"Distribución: {fmt(vh)}")
    ax.grid(alpha=0.3)
    mostrar(fig)

    st.subheader("3.2 Dispersión por especie")
    s1, s2 = st.columns(2)
    sx = s1.selectbox("Eje X", NUM, index=2, format_func=fmt, key="sx")
    sy = s2.selectbox("Eje Y", NUM, index=3, format_func=fmt, key="sy")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for e, g in d.groupby('especies'):
        ax.scatter(g[sx], g[sy], c=COLOR[e], label=e, alpha=0.7, s=50)
    ax.set(xlabel=f"{fmt(sx)} (cm)", ylabel=f"{fmt(sy)} (cm)", title=f"{fmt(sy)} vs. {fmt(sx)}")
    ax.legend()
    ax.grid(alpha=0.3)
    mostrar(fig)

    st.subheader("3.3 Barras: media por especie")
    vb = st.selectbox("Variable", NUM, index=1, format_func=fmt, key="b")
    m = d.groupby('especies')[vb].mean()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(m.index, m.values, color=[COLOR[e] for e in m.index], edgecolor='black')
    ax.set(xlabel="Especie", ylabel=f"{fmt(vb)} promedio (cm)", title=f"Media de {fmt(vb).lower()}")
    ax.grid(axis='y', alpha=0.3)
    mostrar(fig)

    st.subheader("3.4 Subplots: histogramas de todas las características")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, c in zip(axes.flat, NUM):
        ax.hist(d[c], bins=15, color='#4ECDC4', edgecolor='black', alpha=0.8)
        ax.set(xlabel=f"{fmt(c)} (cm)", ylabel="Frecuencia", title=fmt(c))
        ax.grid(alpha=0.3)
    fig.tight_layout()
    mostrar(fig)

# ---------------- Parte 4 ----------------
with tab4:
    st.subheader("1. ¿Qué especie tiene, en promedio, los pétalos más grandes?")
    area = d.groupby('especies')['area_petalo'].mean().sort_values(ascending=False)
    st.dataframe(area.to_frame("Área promedio (cm²)"))
    st.success(f"**Respuesta:** {area.index[0]} (área promedio {area.iloc[0]:.2f} cm²).")

    st.subheader("2. ¿Existe una relación lineal entre el largo y ancho del pétalo?")
    if len(d) > 1:
        r = np.corrcoef(d['largo_petalo'], d['ancho_petalo'])[0, 1]
        st.metric("Coeficiente de correlación", f"{r:.4f}")
        st.success(f"**Respuesta:** correlación {fuerza(r)} (r = {r:.2f}). "
                   "Se aprecia en el gráfico de dispersión de la pestaña 3.")

    st.subheader("3. ¿Qué característica diferencia mejor a las especies?")
    if d['especies'].nunique() > 1:
        prom = d.groupby('especies')[NUM].mean()
        sep_rel = (prom.max() - prom.min()) / d[NUM].std()
        st.dataframe(prom)
        st.markdown("Separación relativa = rango de los promedios ÷ desviación estándar "
                    "(así las variables son comparables entre sí):")
        st.dataframe(sep_rel.sort_values(ascending=False).to_frame("Separación relativa"))
        mejor = sep_rel.idxmax()
        st.success(f"**Respuesta:** «{fmt(mejor)}» es la más útil: es la que más separa los promedios de las especies.")
    else:
        st.info("Selecciona al menos 2 especies en la barra lateral para comparar.")
