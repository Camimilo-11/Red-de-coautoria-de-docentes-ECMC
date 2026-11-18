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

import graphrag as R
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
st.caption("¿Quién colabora con quién, y quién debería colaborar? Predicción de enlaces con GraphSAGE y preguntas "
           "en lenguaje natural con Graph RAG (recuperación en el grafo + respuesta generada por un LLM).")

tab_red, tab_rag, tab_perfil, tab_emb, tab_rec, tab_eval = st.tabs(
    ["🕸️ Red", "💬 Pregúntale a la red", "🧭 Perfil temático", "🗺️ Mapa de embeddings", "🤝 Recomendador",
     "📈 Evaluación"])

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
    actuales = coaut[(coaut["a"] == elegido) | (coaut["b"] == elegido)].copy()
    if len(actuales):
        actuales["Coautor actual"] = [nombre_de[b if a == elegido else a] for a, b in zip(actuales["a"], actuales["b"])]
        actuales["Departamento"] = actuales["Coautor actual"].map(dict(zip(doc["nombre"], doc["departamento"])))
        actuales = actuales.rename(columns={"n_obras": "Obras conjuntas", "primer_anio": "Desde", "ultimo_anio": "Hasta"})
        st.markdown("**Colaboraciones que ya existen** (por eso no aparecen como recomendación):")
        st.dataframe(actuales[["Coautor actual", "Departamento", "Obras conjuntas", "Desde", "Hasta"]]
                     .sort_values("Obras conjuntas", ascending=False), hide_index=True, width="stretch")

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

# ------------------------------------------------------------------ 6. Graph RAG
def secreto(nombre, defecto=None):
    try:
        return st.secrets.get(nombre, defecto)
    except Exception:          # sin archivo de secrets en local
        return defecto


@st.cache_resource
def indice_rag(_datos, _coaut, _pred, _comunidades):
    return R.IndiceGraphRAG(_datos, _coaut, _pred, _comunidades)


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def respuesta_cacheada(pregunta, contexto_txt, proveedor, modelo, _ctx, _clave):
    # La caché se indexa por pregunta + contexto + modelo: repetir una pregunta no vuelve a llamar a la API
    if proveedor == "nvidia":
        r = R.responder_con_nvidia(pregunta, _ctx, _clave, modelo)
    elif proveedor == "openai":
        r = R.responder_con_openai(pregunta, _ctx, _clave, modelo)
    else:
        r = R.responder_con_claude(pregunta, _ctx, _clave, modelo)
    if not r["texto"]:
        raise RuntimeError(r["error"])    # los errores no se guardan en caché: el siguiente intento vuelve a llamar
    return r


EJEMPLOS = [
    "¿Quién de Computación trabaja en visión computacional y ha colaborado con alguien de Matemáticas?",
    "¿Cómo está conectado Hugo Leiva con Diego Peluffo?",
    "¿Con quién debería colaborar Peluffo en el futuro?",
    "¿Qué docentes trabajan en ecuaciones diferenciales y quiénes son sus coautores?",
    "¿Quiénes son los docentes más conectados de la red?",
]


def poner_pregunta(texto):
    st.session_state["pregunta_rag"] = texto


with tab_rag:
    # Proveedor del LLM según la clave que haya en Secrets (prioridad: NVIDIA > OpenAI > Anthropic)
    if secreto("NVIDIA_API_KEY"):
        proveedor, clave = "nvidia", str(secreto("NVIDIA_API_KEY")).strip().removeprefix("Bearer ").strip()
        modelo_llm = secreto("NVIDIA_MODEL", R.NVIDIA_MODELO_POR_DEFECTO)  # con respaldo automático si está retirado
    elif secreto("OPENAI_API_KEY"):
        proveedor, clave = "openai", secreto("OPENAI_API_KEY")
        modelo_llm = secreto("OPENAI_MODEL", R.OPENAI_MODELO_POR_DEFECTO)
    else:
        proveedor, clave = "anthropic", secreto("ANTHROPIC_API_KEY")
        modelo_llm = secreto("CLAUDE_MODEL", R.MODELO_POR_DEFECTO)
    indice = indice_rag(datos, coaut, pred, comunidad_de)
    st.markdown(
        "**Graph RAG.** Escribe una pregunta: la app identifica docentes, departamentos y temas, recorre el grafo "
        "(coautores, caminos y colaboraciones predichas por GraphSAGE), arma un contexto de hechos numerados y "
        "un LLM redacta la respuesta citándolos.")
    st.pills("Ejemplos (clic para usar)", EJEMPLOS, key="ejemplo_rag",
             on_change=lambda: poner_pregunta(st.session_state["ejemplo_rag"] or st.session_state["pregunta_rag"]))
    st.session_state.setdefault("pregunta_rag", EJEMPLOS[0])
    with st.form("form_rag"):
        pregunta = st.text_input("Pregunta", key="pregunta_rag")
        enviar = st.form_submit_button("Preguntar", type="primary")
    if enviar and pregunta.strip():
        st.session_state["consulta_rag"] = pregunta.strip()
    consulta = st.session_state.get("consulta_rag")

    if consulta:
        ctx = indice.recuperar(consulta)
        e = ctx.entidades
        col_resp, col_graf = st.columns([5, 4])
        with col_resp:
            st.markdown(f"**Pregunta:** {consulta}")
            if clave:
                with st.spinner(f"Consultando al LLM ({modelo_llm})…"):
                    try:
                        r = respuesta_cacheada(consulta, ctx.texto(), proveedor, modelo_llm, ctx, clave)
                    except RuntimeError as ex:
                        r = {"texto": None, "error": str(ex)}
                if r["texto"]:
                    st.markdown(r["texto"])
                    st.caption(f"Generado por {r.get('modelo', modelo_llm)} a partir de {len(ctx.hechos)} hechos "
                               "recuperados del grafo. Los números entre corchetes remiten al contexto (abajo).")
                else:
                    st.error(r["error"])
            else:
                st.info("Modo sin LLM: falta `NVIDIA_API_KEY`, `OPENAI_API_KEY` o `ANTHROPIC_API_KEY` en los *Secrets*. Se "
                        "muestran los hechos más relevantes que recuperó el grafo; con la clave, el LLM los convierte "
                        "en una respuesta.")
                relevantes = [h for h in ctx.hechos[1:] if "[semilla]" in h or h.startswith(("Camino", "Predicción"))]
                for h in relevantes[:10]:
                    st.markdown(f"- {h.replace('; [semilla].', '.')}")
            chips = ([f"👤 {nombre_de[d]}" for d in e["docentes"]] + [f"🏛️ {d}" for d in e["departamentos"]]
                     + [f"🏷️ {indice.sub_nombre[s]}" for s in e["sublineas"][:6]]
                     + [f"🎯 {i}" for i in e["intenciones"]])
            st.caption("Entidades detectadas: " + (" · ".join(chips) if chips else "ninguna (se usan los docentes más conectados)"))
        with col_graf:
            semillas = set(ctx.semillas)
            en_camino = {tuple(sorted(p)) for c in ctx.caminos if c[1] is not None for p in zip(c, c[1:])}
            sub = Network(height="460px", width="100%", bgcolor="#fcfcfb", font_color="#0b0b0b", cdn_resources="in_line")
            for d in ctx.nodos:
                sub.add_node(d, label=corto_de[d], color=COLOR_DEP[dep_de[d]],
                             size=24 if d in semillas else 12, borderWidth=4 if d in semillas else 1,
                             title=f"{nombre_de[d]}\n{dep_de[d]}\n{'Semilla (relevante para la pregunta)' if d in semillas else 'Vecino en la red'}")
            for a, b in ctx.aristas:
                camino = tuple(sorted((a, b))) in en_camino
                sub.add_edge(a, b, color=TINTA if camino else GRIS, width=5 if camino else 1.5)
            for a, b, p in ctx.aristas_pred:
                sub.add_edge(a, b, dashes=True, width=3, color="#6250d6", title=f"Predicha · p = {p:.2f}")
            sub.set_options("""{"nodes": {"font": {"size": 13, "color": "#52514e", "strokeWidth": 3, "strokeColor": "#fcfcfb"},
                                          "color": {"border": "#0b0b0b"}, "scaling": {"label": {"drawThreshold": 1}}},
                                "edges": {"smooth": false},
                                "physics": {"barnesHut": {"gravitationalConstant": -2500, "springLength": 90}}}""")
            components.html(sub.generate_html(), height=480)
            st.caption("Subgrafo recuperado · nodos grandes con borde = semillas · línea gruesa oscura = camino entre "
                       "docentes nombrados · punteado violeta = colaboración predicha.")
        with st.expander(f"Contexto recuperado que recibe el LLM ({len(ctx.hechos)} hechos)"):
            st.text(ctx.texto())
