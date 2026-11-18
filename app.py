"""Red de coautoría de docentes ECMC + GraphSAGE para predecir colaboraciones.

Ejecutar:  streamlit run app.py
"""
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
from sklearn.manifold import TSNE

import modelo as M

RAIZ = Path(__file__).parent
COLOR_DEP = {"Matemáticas": "#2a78d6", "Computación": "#eb6834", "Desarrollo Humano": "#1baf7a"}
COLORES_SERIE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GRIS = "#9a9893"
TINTA = "#52514e"

st.set_page_config(page_title="Red de coautoría ECMC · GraphSAGE", page_icon="🕸️", layout="wide")


# ------------------------------------------------------------------ carga
@st.cache_resource
def cargar_modelo():
    with open(RAIZ / "model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def cargar():
    datos = M.cargar_datos(RAIZ / "data")
    coaut = M.coautorias(datos)
    return datos, coaut


def nombre_corto(nombre: str) -> str:
    t = nombre.split()
    if len(t) >= 4:
        return f"{t[0]} {t[2]}"
    if len(t) == 3:
        return f"{t[0]} {t[1]}"
    return nombre


@st.cache_data
def inferencia(_paquete, corte_final):
    """Forward de GraphSAGE con los pesos de model.pkl (NumPy puro)."""
    datos, coaut = cargar()
    nodos = _paquete["nodos"]
    n = len(nodos)
    X, _ = M.atributos(datos, corte_final, tuple(_paquete["stats_pub_final"]))
    ar = M.aristas_hasta(coaut, nodos, corte_final)
    A = M.matriz_media(ar, n)
    conectados = {tuple(sorted(e)) for e in ar.tolist()}
    cand = np.array([(i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in conectados])
    prob = M.predecir(_paquete["params_final"], X, A, cand)
    Z = M.embeddings(_paquete["params_final"][0], X, A)
    con_obras = (datos["part"].groupby("id_docente")["id_obra"].nunique()
                 .reindex(nodos, fill_value=0).to_numpy() > 0)
    xy = np.full((n, 2), np.nan)   # t-SNE solo con docentes que tienen producción formal
    xy[con_obras] = TSNE(n_components=2, perplexity=10, random_state=0, init="pca").fit_transform(Z[con_obras])
    return cand, prob, xy


paquete = cargar_modelo()
datos, coaut = cargar()
doc = datos["docentes"].copy()
doc["corto"] = doc["nombre"].map(nombre_corto)
idx = {d: i for i, d in enumerate(paquete["nodos"])}
nombre_de = dict(zip(doc["id"], doc["nombre"]))
corto_de = dict(zip(doc["id"], doc["corto"]))
dep_de = dict(zip(doc["id"], doc["departamento"]))
n_obras = datos["part"].groupby("id_docente")["id_obra"].nunique().reindex(doc["id"], fill_value=0)
corte_final = paquete["corte_final"]
cand, prob, xy = inferencia(paquete, corte_final)

pred = pd.DataFrame({"a": [paquete["nodos"][i] for i in cand[:, 0]],
                     "b": [paquete["nodos"][j] for j in cand[:, 1]], "prob": prob})
# Docentes sin producción formal: sus atributos son idénticos (todo cero), así que el modelo
# no tiene información para ellos. Se excluyen de las predicciones.
sin_prod = set(n_obras[n_obras == 0].index)
pred = pred[~pred["a"].isin(sin_prod) & ~pred["b"].isin(sin_prod)]
pred = pred.sort_values("prob", ascending=False).reset_index(drop=True)

G_total = nx.Graph()
G_total.add_nodes_from(doc["id"])
G_total.add_edges_from(zip(coaut["a"], coaut["b"]))
comunidades = nx.community.louvain_communities(G_total, seed=0)
comunidad_de = {}
for k, c in enumerate(sorted(comunidades, key=len, reverse=True)):
    for d in c:
        comunidad_de[d] = f"C{k + 1}" if len(c) > 1 else "Sin coautores"

# ------------------------------------------------------------------ barra lateral
st.sidebar.title("Filtros")
deps = st.sidebar.multiselect("Departamentos", M.DEPARTAMENTOS, default=M.DEPARTAMENTOS)
anio_min, anio_max = int(coaut["primer_anio"].min()), int(coaut["primer_anio"].max())
anio = st.sidebar.slider("Mostrar coautorías iniciadas hasta", anio_min, anio_max, anio_max)
top_k = st.sidebar.slider("Colaboraciones predichas a dibujar", 0, 30, 10)
ocultar = st.sidebar.checkbox("Ocultar docentes sin coautores (en la red)", value=False)
st.sidebar.caption(
    f"Datos: consolidado ECMC, solo producción formal · {len(doc)} docentes · "
    f"modelo entrenado {paquete['fecha_entrenamiento']}")

sel = doc[doc["departamento"].isin(deps)]["id"].tolist()
sel_set = set(sel)

st.title("Red de coautoría de docentes ECMC")
st.caption("¿Quién colabora con quién, y quién debería colaborar? Predicción de enlaces con GraphSAGE, "
           "que razona sobre vecinos a 2 saltos: el mismo principio que usa un Graph RAG (p. ej. GNN-RAG) "
           "para recuperar información por relaciones.")

tab_red, tab_perfil, tab_emb, tab_rec, tab_eval = st.tabs(
    ["🕸️ Red", "🧭 Perfil temático", "🗺️ Mapa de embeddings", "🤝 Recomendador", "📈 Evaluación"])

# ------------------------------------------------------------------ 1. red
with tab_red:
    vis = coaut[(coaut["primer_anio"] <= anio) & coaut["a"].isin(sel_set) & coaut["b"].isin(sel_set)]
    G = nx.Graph()
    G.add_nodes_from(sel)
    G.add_edges_from(zip(vis["a"], vis["b"]))
    inter = (vis["a"].map(dep_de) != vis["b"].map(dep_de)).sum()
    mayor = max((len(c) for c in nx.connected_components(G)), default=0)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Docentes", G.number_of_nodes())
    c2.metric("Pares de coautores", G.number_of_edges())
    c3.metric("Interdepartamentales", int(inter))
    c4.metric("Sin coautores", nx.number_of_isolates(G))
    c5.metric("Componente mayor", mayor)

    p_vis = pred[pred["a"].isin(sel_set) & pred["b"].isin(sel_set)].head(top_k)
    nodos_red = sel
    if ocultar:
        con_arista = set(vis["a"]) | set(vis["b"]) | set(p_vis["a"]) | set(p_vis["b"])
        nodos_red = [d for d in sel if d in con_arista]
    net = Network(height="640px", width="100%", bgcolor="#fcfcfb", font_color="#0b0b0b", cdn_resources="in_line")
    for d in nodos_red:
        net.add_node(d, label=corto_de[d], color=COLOR_DEP[dep_de[d]],
                     size=float(10 + 2.5 * np.sqrt(n_obras[d])),
                     title=f"{nombre_de[d]}\n{dep_de[d]}\nObras formales: {n_obras[d]}\nComunidad: {comunidad_de[d]}")
    for _, r in vis.iterrows():
        net.add_edge(r["a"], r["b"], width=1 + np.log1p(r["n_obras"]) * 1.5, color=GRIS,
                     title=f"{r['n_obras']} obra(s) · {r['primer_anio']}–{r['ultimo_anio']}")
    for _, r in p_vis.iterrows():
        net.add_edge(r["a"], r["b"], dashes=True, width=3, color="#6250d6",
                     title=f"Predicha por GraphSAGE · p = {r['prob']:.2f}")
    net.set_options("""{
      "nodes": {"font": {"size": 15, "color": "#52514e", "strokeWidth": 3, "strokeColor": "#fcfcfb"},
                "borderWidth": 2, "color": {"border": "#fcfcfb"},
                "scaling": {"label": {"drawThreshold": 1}}},
      "edges": {"smooth": false},
      "physics": {"barnesHut": {"gravitationalConstant": -3500, "springLength": 110, "avoidOverlap": 0.4},
                  "stabilization": {"iterations": 400}},
      "interaction": {"hover": true, "tooltipDelay": 80}
    }""")
    components.html(net.generate_html(), height=660)
    st.caption("Color = departamento · tamaño = n.º de obras formales · grosor = obras compartidas · "
               "línea punteada violeta = colaboración predicha (aún no existe). Arrastra los nodos y pasa el cursor.")

# ------------------------------------------------------------------ 2. perfil temático
with tab_perfil:
    part = datos["part"].merge(datos["obra_sub"], on="id_obra")
    lineas = datos["lineas"].drop_duplicates("id_linea").set_index("id_linea")["linea"]
    tabla = (part.drop_duplicates(["id_docente", "id_obra", "id_linea"])
             .pipe(lambda d: pd.crosstab(d["id_docente"], d["id_linea"]))
             .reindex(index=sel, fill_value=0))
    prop = tabla.div(tabla.sum(axis=1).replace(0, 1), axis=0)
    orden = sorted(sel, key=lambda d: (M.DEPARTAMENTOS.index(dep_de[d]), corto_de[d]))
    prop = prop.reindex(orden)
    corta = {"math": "Matemáticas y<br>optimización", "data": "Ciencia de<br>datos", "scientific": "Computación<br>científica",
             "ai": "Inteligencia<br>artificial", "it": "Tecnologías de<br>la información", "society": "Ciencia, tecnología<br>y sociedad"}
    fig = go.Figure(go.Heatmap(
        z=prop.values, x=[corta.get(c, c) for c in prop.columns],
        customdata=np.tile([lineas.get(c, c) for c in prop.columns], (len(prop), 1)),
        y=[f"{corto_de[d]} · {dep_de[d][:4]}." for d in prop.index],
        colorscale="Blues", zmin=0, zmax=1, colorbar=dict(title="Proporción"),
        hovertemplate="%{y}<br>%{customdata}<br>%{z:.0%} de sus obras<extra></extra>", xgap=2, ygap=2))
    fig.update_layout(height=max(400, 22 * len(prop)), margin=dict(l=10, r=10, t=30, b=10),
                      xaxis=dict(side="top", tickangle=0, tickfont=dict(size=11)),
                      yaxis=dict(autorange="reversed"), plot_bgcolor="#fcfcfb")
    st.plotly_chart(fig, width="stretch")
    st.caption("Proporción de obras formales de cada docente por línea de investigación de la Universidad "
               "(una obra puede tener varias líneas). Estos perfiles son los atributos de entrada del modelo.")

# ------------------------------------------------------------------ 3. embeddings
with tab_emb:
    emb = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "id": paquete["nodos"]})
    emb["Docente"] = emb["id"].map(nombre_de)
    emb["Departamento"] = emb["id"].map(dep_de)
    emb["Comunidad"] = emb["id"].map(comunidad_de)
    emb["Obras"] = emb["id"].map(n_obras)
    emb = emb[emb["id"].isin(sel_set) & ~emb["id"].isin(sin_prod)]
    fig = px.scatter(emb, x="x", y="y", color="Departamento", color_discrete_map=COLOR_DEP,
                     hover_name="Docente", hover_data={"x": False, "y": False, "Comunidad": True, "Obras": True},
                     category_orders={"Departamento": M.DEPARTAMENTOS})
    fig.update_traces(marker=dict(size=14, line=dict(width=2, color="#fcfcfb")))
    fig.update_layout(height=620, xaxis=dict(visible=False), yaxis=dict(visible=False),
                      plot_bgcolor="#fcfcfb", legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig, width="stretch")
    st.caption("Proyección t-SNE de los embeddings de 16 dimensiones que aprende GraphSAGE. Docentes cercanos "
               "tienen vecindarios y perfiles temáticos parecidos: el modelo los considera candidatos a colaborar. "
               "Pasa el cursor sobre un punto para ver el nombre. "
               f"Se omiten {len(sin_prod)} docentes sin producción formal (sin atributos para el modelo).")

# ------------------------------------------------------------------ 4. recomendador
with tab_rec:
    opciones = sorted([d for d in sel if d not in sin_prod], key=lambda d: nombre_de[d])
    if sin_prod:
        st.info(f"{len(sin_prod)} docentes no tienen producción formal registrada; el modelo no tiene datos "
                "para recomendarles colaboradores y no aparecen aquí.")
    elegido = st.selectbox("Docente", opciones, format_func=lambda d: f"{nombre_de[d]} ({dep_de[d]})")
    perfil = M.perfil_tematico(datos, corte_final)
    sub_nombre = datos["lineas"].set_index("id_sublinea")["sublinea"]
    filas = pred[(pred["a"] == elegido) | (pred["b"] == elegido)].head(5)
    recs = []
    for _, r in filas.iterrows():
        otro = r["b"] if r["a"] == elegido else r["a"]
        comun = (perfil.loc[elegido] * perfil.loc[otro]).sort_values(ascending=False)
        comun = [sub_nombre[s] for s in comun[comun > 0].index[:2]]
        try:
            camino = nx.shortest_path(G_total, elegido, otro)
            camino_txt = " → ".join(corto_de[c] for c in camino)
        except nx.NetworkXNoPath:
            camino_txt = "Sin camino en la red (recomendación por afinidad temática)"
        recs.append({"Docente sugerido": nombre_de[otro], "Departamento": dep_de[otro],
                     "Probabilidad": r["prob"], "Sublíneas en común": "; ".join(comun) or "—",
                     "Camino actual en la red": camino_txt})
    recs = pd.DataFrame(recs)
    col_a, col_b = st.columns([2, 3])
    with col_a:
        fig = go.Figure(go.Bar(x=recs["Probabilidad"], y=recs["Docente sugerido"].map(nombre_corto), orientation="h",
                               marker=dict(color=[COLOR_DEP[d] for d in recs["Departamento"]], cornerradius=4),
                               text=[f"{p:.2f}" for p in recs["Probabilidad"]], textposition="outside",
                               hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
        fig.update_layout(height=320, xaxis=dict(range=[0, 1.1], title="Probabilidad predicha"),
                          yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=10, b=10),
                          plot_bgcolor="#fcfcfb", bargap=0.35)
        st.plotly_chart(fig, width="stretch")
    with col_b:
        st.dataframe(recs, hide_index=True, width="stretch",
                     column_config={"Probabilidad": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1)})
    st.caption(f"Top 5 docentes con los que {nombre_de[elegido]} aún no ha publicado, según el ensamble de "
               f"10 modelos entrenados con toda la red hasta {corte_final}. Color de barra = departamento.")

# ------------------------------------------------------------------ 5. evaluación
with tab_eval:
    ev = paquete["evaluacion"]
    corte = paquete["corte_evaluacion"]
    st.markdown(
        f"**Protocolo temporal.** Se entrena con las coautorías iniciadas hasta **{corte}** (atributos calculados "
        f"solo con obras hasta {corte}) y se predice cuáles de los **{ev['n_candidatos']}** pares aún no "
        f"conectados empiezan a colaborar entre {corte + 1} y {corte_final}: **{ev['n_positivos']}** lo hicieron. "
        f"El subconjunto *sin historial* son los pares donde al menos un docente no tenía coautores en {corte} "
        f"({ev['n_candidatos_frio']} pares, {ev['n_positivos_frio']} positivos). "
        "IC 95 % por bootstrap pareado (1000 remuestreos).")
    vista = st.radio("Subconjunto", ["global", "frio"], horizontal=True,
                     format_func=lambda v: "Todos los pares" if v == "global" else "Docentes sin historial de coautoría")
    curvas = ev["curvas"][vista]
    fig = go.Figure()
    for k, (nombre, c) in enumerate(curvas.items()):
        fig.add_trace(go.Scatter(x=c["fpr"], y=c["tpr"], mode="lines", name=f"{nombre} (AUC {c['auc']:.2f})",
                                 line=dict(width=3 if k == 0 else 2, color=COLORES_SERIE[k])))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar", line=dict(dash="dot", color=GRIS, width=1)))
    fig.update_layout(height=480, xaxis_title="Tasa de falsos positivos", yaxis_title="Tasa de verdaderos positivos",
                      plot_bgcolor="#fcfcfb", hovermode="x unified", legend=dict(y=0.02, x=0.55))
    st.plotly_chart(fig, width="stretch")

    tabla = pd.DataFrame([{
        "Método": nombre, "AUC-ROC": round(c["auc"], 3),
        "IC 95 %": f"[{c['ic95'][0]:.3f}, {c['ic95'][1]:.3f}]",
        "Precisión media (AP)": round(c["ap"], 3),
        "Δ AUC GraphSAGE − método (IC 95 %)": "—" if k == 0 else f"[{c['dif_ic95'][0]:+.3f}, {c['dif_ic95'][1]:+.3f}]",
    } for k, (nombre, c) in enumerate(curvas.items())])
    st.dataframe(tabla, hide_index=True, width="stretch")
    tasa = ev["n_positivos"] / ev["n_candidatos"] if vista == "global" else ev["n_positivos_frio"] / ev["n_candidatos_frio"]
    st.caption(f"Línea base de AP (clasificador al azar) = {tasa:.3f}. Un IC de la diferencia que no incluye 0 "
               "indica una mejora de GraphSAGE estadísticamente distinguible en esta muestra.")

    ps = pd.DataFrame(ev["por_semilla"])
    col = "auc" if vista == "global" else "auc_frio"
    st.markdown(f"**Variabilidad entre semillas (modelos individuales):** AUC = {ps[col].mean():.3f} ± "
                f"{ps[col].std():.3f} (media ± DE, n = {len(ps)}). El ensamble promedia las 10 predicciones.")

    with st.expander("Cómo funciona el modelo"):
        st.markdown(
            "- **Nodos:** 50 docentes. **Aristas:** haber publicado al menos una obra formal juntos.\n"
            "- **Atributos de cada docente:** departamento (one-hot), log del n.º de obras y proporción de obras "
            "en cada una de las 46 sublíneas de investigación.\n"
            "- **GraphSAGE (media), 2 capas:** cada docente combina sus atributos con el promedio de los de sus "
            "coautores; con dos capas también incorpora a los coautores de sus coautores (2 saltos).\n"
            "- **Decodificador:** MLP sobre [z_u ⊙ z_v, |z_u − z_v|] → probabilidad de colaborar.\n"
            "- **Entrenamiento:** en cada época se ocultan el 30 % de las aristas del paso de mensajes y se usan "
            "como positivos, con 3 negativos por positivo; Adam, 300 épocas, dropout 0.3. "
            "Hiperparámetros fijados a priori.\n"
            "- **Por qué GraphSAGE:** es inductivo; un docente sin coautores igual obtiene un embedding a partir "
            "de sus atributos, mientras que las heurísticas estructurales le asignan puntaje 0.\n"
            "- **Limitaciones:** red pequeña (33 positivos de prueba, 12 en el subconjunto sin historial), "
            "los perfiles temáticos se infieren de los títulos y 7 docentes no tienen producción formal.")
