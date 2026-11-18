"""Graph RAG sobre la red de coautoría de docentes ECMC.

Flujo (recuperación local estilo GraphRAG):
  1. Enlace de entidades: docentes nombrados, departamentos, sublíneas
     (diccionario de términos ES/EN) y obras (TF-IDF de n-gramas de caracteres
     sobre los títulos).
  2. Semillas: docentes más relevantes para esas entidades.
  3. Expansión en el grafo: vecinos a 1 salto (2 si se pregunta por caminos),
     caminos más cortos entre docentes nombrados y, si se pide, colaboraciones
     predichas por GraphSAGE.
  4. Contexto: hechos numerados [n] que se entregan al LLM.
  5. Generación: Claude redacta la respuesta citando los hechos. Sin clave de
     API, la app muestra el contexto recuperado (modo extractivo).
"""
import re
import unicodedata
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

MODELO_POR_DEFECTO = "claude-opus-5-5"
MODELOS_CON_FALLBACK = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}

# Términos (sin tildes, en minúscula) que activan cada sublínea. ES + EN porque los títulos están en inglés.
TERMINOS_SUBLINEA = {
    "math.differential": ["ecuaciones diferenciales", "ecuacion diferencial", "differential equation", "edp", "pde"],
    "math.analysis": ["analisis matematico", "functional analysis", "analisis funcional"],
    "math.control": ["teoria de control", "controlabilidad", "controllab", "sincronizacion", "synchroniz"],
    "math.dynamics": ["sistemas dinamicos", "dynamical system", "caos", "chaos", "bifurcac"],
    "math.probability": ["probabilidad", "estadistic", "statistic", "probabil", "estocastic", "stochastic"],
    "math.algebra": ["algebra", "geometria", "geometr"],
    "math.topology": ["topolog"],
    "math.discrete": ["matematica discreta", "logica", "teoria de grafos", "graph theory", "combinator"],
    "math.complex": ["analisis complejo", "variable compleja", "complex analysis"],
    "math.matheducation": ["didactica", "educacion matematica", "ensenanza de la matematica", "mathematics education"],
    "data.analysis": ["analisis de datos", "data analysis"],
    "data.databases": ["base de datos", "bases de datos", "database", "sql"],
    "data.multivariate": ["multivariante", "multivariate"],
    "data.computationalstats": ["estadistica computacional", "computational statistic", "bayes"],
    "data.mining": ["mineria de datos", "data mining"],
    "data.visualization": ["visualizacion de datos", "data visualization"],
    "data.ethics": ["privacidad", "privacy"],
    "scientific.simulation": ["simulacion", "simulation", "modelamiento", "modelado"],
    "scientific.hpc": ["alto rendimiento", "hpc", "paralel", "parallel", "gpu"],
    "scientific.algorithms": ["algoritmo", "algorithm"],
    "scientific.numerical": ["analisis numerico", "metodos numericos", "numerical"],
    "scientific.theory": ["teoria de la computacion", "computabilidad", "automata"],
    "scientific.discrete": ["estructuras discretas"],
    "scientific.visualization": ["visualizacion cientifica", "scientific visualization"],
    "scientific.iot": ["iot", "internet de las cosas", "internet of things", "sensor"],
    "ai.ml": ["aprendizaje automatico", "machine learning", "clasificacion", "classification"],
    "ai.deep": ["aprendizaje profundo", "deep learning", "redes neuronales", "red neuronal", "neural network", "cnn", "transformer"],
    "ai.vision": ["vision computacional", "vision por computador", "computer vision", "imagen", "image", "segmentacion"],
    "ai.llm": ["llm", "modelos de lenguaje", "modelo de lenguaje", "language model", "procesamiento de lenguaje", "nlp"],
    "ai.evolution": ["evolutiv", "genetic", "genetico", "metaheurist"],
    "ai.business": ["negocio", "business"],
    "it.transport": ["transporte", "vehicular", "vanet", "v2x", "trafico", "traffic"],
    "it.security": ["seguridad informatica", "ciberseguridad", "security", "cyber", "criptograf", "phishing", "malware"],
    "it.networks": ["redes de comunicacion", "telecomunicac", "wireless", "inalambric", "5g", "6g", "wifi", "wi fi"],
    "it.software": ["software"],
    "it.vr": ["realidad virtual", "realidad aumentada", "virtual reality", "augmented reality"],
    "it.cloud": ["la nube", "computacion en la nube", "cloud"],
    "it.hci": ["interaccion humano", "human computer", "usabilidad", "usability"],
    "society.history": ["historia de la ciencia", "filosofia", "philosophy"],
    "society.education": ["educacion", "educativ", "ensenanza", "pedagog", "education"],
    "society.ethics": ["etica", "ethic"],
    "society.enterprise": ["empresa", "emprendimiento", "entrepreneur"],
    "society.innovation": ["innovacion", "tecnologias emergentes", "innovation"],
    "society.opinion": ["opinion publica", "redes sociales", "social media", "social network"],
    "society.communication": ["publicidad", "relaciones publicas", "marketing"],
    "society.convergence": ["interdisciplinar"],
}
TERMINOS_LINEA = {
    "ai": ["inteligencia artificial", " ia ", "artificial intelligence"],
    "data": ["ciencia de datos", "data science"],
    "scientific": ["computacion cientifica", "scientific computing"],
    "it": ["tecnologias de la informacion"],
}
DEPARTAMENTOS = {"Matemáticas": ["matematic"], "Computación": ["computacion", "computer science"],
                 "Desarrollo Humano": ["desarrollo humano"]}
INTENCION = {
    "camino": ["camino", "conect", "relacion entre", "relacionad", "puente", "a traves de", "llegar", "distancia"],
    "prediccion": ["deberia", "podria", "podrian", "recomend", "futur", "potencial", "predic", "probable", "sugier", "sugerenc"],
    "central": ["mas conectad", "central", "mas colabor", "mas coautor", "mas importante", "hub", "clave en la red"],
    "comunidad": ["comunidad", "comunidades", "grupo", "grupos", "cluster"],
    "interdep": ["interdepartamental", "entre departamentos", "otro departamento", "distinto departamento"],
}
STOP = set("""a al algo alguien algun alguna ante con contra cual cuales cuando de del desde donde el ella ellos en entre es
esta este esto estos han has hay la las le les lo los mas me mi mis muy no nos o para pero por que quien quienes se
segun ser si sin sobre su sus tambien te tiene tienen trabaja trabajan trabajado un una uno unos y ya yo docente docentes
profesor profesores profesora investigador investigadores colabora colaboran colaborado colaboracion coautor coautores
publicado publicaron publicacion publicaciones obra obras paper papers articulo articulos red ecmc escuela area areas tema
temas linea lineas investigacion sobre hace han hecho cuantos cuantas cual cuales""".split())


def normalizar(texto: str) -> str:
    t = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode().lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return " " + t.strip() + " "


@dataclass
class Contexto:
    pregunta: str
    entidades: dict
    semillas: list
    nodos: list
    aristas: list                     # (a, b) coautorías reales dentro del subgrafo
    aristas_pred: list                # (a, b, p) colaboraciones predichas
    caminos: list = field(default_factory=list)
    hechos: list = field(default_factory=list)

    def texto(self) -> str:
        return "\n".join(f"[{i}] {h}" for i, h in enumerate(self.hechos, 1))


class IndiceGraphRAG:
    def __init__(self, datos: dict, coaut: pd.DataFrame, pred: pd.DataFrame, comunidad_de: dict):
        self.doc = datos["docentes"].set_index("id")
        self.part = datos["part"]
        self.obras = datos["obras"].set_index("id_obra")
        self.sub_nombre = datos["lineas"].set_index("id_sublinea")["sublinea"].to_dict()
        self.sub_linea = datos["lineas"].set_index("id_sublinea")["id_linea"].to_dict()
        self.coaut = coaut
        self.pred = pred
        self.comunidad_de = comunidad_de
        self.G = nx.Graph()
        self.G.add_nodes_from(self.doc.index)
        for _, r in coaut.iterrows():
            self.G.add_edge(r["a"], r["b"], n=int(r["n_obras"]), ini=int(r["primer_anio"]), fin=int(r["ultimo_anio"]))
        m = self.part.merge(datos["obra_sub"], on="id_obra")
        tabla = pd.crosstab(m["id_docente"], m["id_sublinea"]).reindex(index=self.doc.index, fill_value=0)
        self.perfil = tabla.div(tabla.sum(axis=1).replace(0, 1), axis=0)
        self.n_obras = self.part.groupby("id_docente")["id_obra"].nunique().reindex(self.doc.index, fill_value=0)
        self.autores = self.part.groupby("id_obra")["id_docente"].apply(lambda s: sorted(set(s))).to_dict()
        # Índice léxico de obras: n-gramas de caracteres toleran ES↔EN (simulación/simulation)
        self.tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 5), sublinear_tf=True, min_df=2)
        self.M_obras = self.tfidf.fit_transform(self.obras["titulo"].map(normalizar))
        self.ids_obras = self.obras.index.to_numpy()
        # Nombres para enlazar docentes
        self.tokens_doc = {}
        for d, nombre in self.doc["nombre"].items():
            t = normalizar(nombre).split()
            self.tokens_doc[d] = {"nombres": t[:2] if len(t) >= 4 else t[:1], "apellidos": t[2:] if len(t) >= 4 else t[1:],
                                  "original": nombre.split()}

    # ------------------------------------------------------------ 1. enlace de entidades
    def enlazar(self, pregunta: str) -> dict:
        q = normalizar(pregunta)
        palabras = set(q.split())
        mayus = {normalizar(w).strip() for w in re.findall(r"\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+", pregunta)}
        nombrados = []
        conteo_apellido = {}
        for d, t in self.tokens_doc.items():
            for ap in t["apellidos"]:
                conteo_apellido[ap] = conteo_apellido.get(ap, 0) + 1
        for d, t in self.tokens_doc.items():
            nom = any(n in palabras for n in t["nombres"])
            aps = [a for a in t["apellidos"] if a in palabras]
            unico = any(a in mayus and conteo_apellido[a] == 1 and len(a) >= 4 for a in aps)
            if (nom and aps) or len(aps) >= 2 or unico:
                nombrados.append(d)
        deps = [dep for dep, ts in DEPARTAMENTOS.items() if any(t in q for t in ts)]
        subs = {s for s, ts in TERMINOS_SUBLINEA.items() if any(t in q for t in ts)}
        for linea, ts in TERMINOS_LINEA.items():
            if any(t in q for t in ts):
                subs |= {s for s, l in self.sub_linea.items() if l == linea}
        intenciones = [k for k, ts in INTENCION.items() if any(t in q for t in ts)]
        # Texto temático para TF-IDF: sin nombres, departamentos ni palabras vacías
        quitar = set(STOP)
        for d in nombrados:
            quitar |= set(self.tokens_doc[d]["nombres"]) | set(self.tokens_doc[d]["apellidos"])
        tema = " ".join(w for w in q.split() if w not in quitar and len(w) > 2 and not w.startswith("matematic")
                        and not w.startswith("computaci"))
        obras = []
        solo_estructura = not subs and not nombrados and set(intenciones) & {"comunidad", "central"}
        if tema.strip() and not solo_estructura:
            sims = (self.M_obras @ self.tfidf.transform([normalizar(tema)]).T).toarray().ravel()
            top = np.argsort(-sims)[:12]
            obras = [(self.ids_obras[i], float(sims[i])) for i in top if sims[i] >= 0.20]
        return {"docentes": nombrados, "departamentos": deps, "sublineas": sorted(subs),
                "obras": obras, "intenciones": intenciones}

    # ------------------------------------------------------------ 2-4. recuperación y contexto
    def recuperar(self, pregunta: str, max_semillas: int = 8, max_nodos: int = 20) -> Contexto:
        e = self.enlazar(pregunta)
        puntaje = pd.Series(0.0, index=self.doc.index)
        if e["sublineas"]:
            puntaje += self.perfil.reindex(columns=e["sublineas"], fill_value=0).sum(axis=1)
        for obra, sim in e["obras"]:
            for d in self.autores.get(obra, []):
                puntaje[d] += 0.5 * sim
        semillas = list(e["docentes"])
        candidatos = puntaje[puntaje > 0].sort_values(ascending=False)
        if e["departamentos"] and len(candidatos):
            for dep in e["departamentos"]:   # garantizar representantes de cada departamento mencionado
                del_dep = [d for d in candidatos.index if self.doc.at[d, "departamento"] == dep][:3]
                semillas += [d for d in del_dep if d not in semillas]
        semillas += [d for d in candidatos.index if d not in semillas][: max(0, max_semillas - len(semillas))]
        if not semillas and e["departamentos"]:
            for dep in e["departamentos"]:
                del_dep = self.n_obras[self.doc["departamento"] == dep].sort_values(ascending=False)
                semillas += list(del_dep.index[:5])
        if not semillas or "central" in e["intenciones"]:
            grado = pd.Series(dict(self.G.degree())).sort_values(ascending=False)
            semillas += [d for d in grado.index[:6] if d not in semillas]

        # Expansión: vecinos a 1 salto de las semillas (los caminos multi-salto se añaden abajo)
        nodos = list(dict.fromkeys(semillas))
        nueva = []
        for d in semillas:
            vecinos = sorted(self.G.neighbors(d), key=lambda v: -self.G[d][v]["n"])
            nueva += [v for v in vecinos if v not in nodos and v not in nueva]
        nodos += nueva[: max(0, max_nodos - len(nodos))]
        caminos = []
        if len(e["docentes"]) >= 2:
            for i, a in enumerate(e["docentes"]):
                for b in e["docentes"][i + 1:]:
                    try:
                        c = nx.shortest_path(self.G, a, b)
                        caminos.append(c)
                        nodos += [x for x in c if x not in nodos]
                    except nx.NetworkXNoPath:
                        caminos.append([a, None, b])
        S = set(nodos)
        en_camino = {tuple(sorted(p)) for c in caminos if c[1] is not None for p in zip(c, c[1:])}
        sem = set(semillas)
        aristas = [(a, b) for a, b in self.G.subgraph(nodos).edges()
                   if a in sem or b in sem or tuple(sorted((a, b))) in en_camino]
        aristas_pred = []
        if "prediccion" in e["intenciones"]:
            p = self.pred[self.pred["a"].isin(semillas) | self.pred["b"].isin(semillas)].head(8)
            aristas_pred = list(zip(p["a"], p["b"], p["prob"]))
            for a, b, _ in aristas_pred:
                for x in (a, b):
                    if x not in S:
                        nodos.append(x)
                        S.add(x)

        ctx = Contexto(pregunta, e, semillas, nodos, aristas, aristas_pred, caminos)
        self._hechos(ctx, caminos)
        return ctx

    def _nombre(self, d):
        return f"{self.doc.at[d, 'nombre']} ({self.doc.at[d, 'departamento']})"

    def _hechos(self, ctx: Contexto, caminos: list):
        h = ctx.hechos
        n_inter = sum(self.doc.at[a, "departamento"] != self.doc.at[b, "departamento"] for a, b in self.G.edges())
        h.append(f"Red completa: {self.G.number_of_nodes()} docentes de la ECMC (Matemáticas, Computación, Desarrollo "
                 f"Humano), {self.G.number_of_edges()} pares de coautores ({n_inter} interdepartamentales), "
                 f"{nx.number_of_isolates(self.G)} docentes sin coautores dentro de la Escuela. Solo producción formal.")
        for d in ctx.nodos:
            fila = self.perfil.loc[d]
            top = fila[fila > 0].sort_values(ascending=False).head(3)
            temas = ", ".join(f"{self.sub_nombre[s]} ({v:.0%})" for s, v in top.items()) or "sin sublíneas registradas"
            rol = "semilla" if d in ctx.semillas else "vecino en la red"
            h.append(f"Docente {self._nombre(d)}: {self.n_obras[d]} obras formales; {self.G.degree(d)} coautores en la "
                     f"Escuela; sublíneas principales: {temas}; comunidad {self.comunidad_de.get(d, '—')}; [{rol}].")
        for a, b in ctx.aristas:
            ed = self.G[a][b]
            inter = " (interdepartamental)" if self.doc.at[a, "departamento"] != self.doc.at[b, "departamento"] else ""
            comunes = [o for o in self.autores if a in self.autores[o] and b in self.autores[o]]
            ej = self.obras.loc[[o for o in comunes if o in self.obras.index]].sort_values("anio", ascending=False).head(1)
            ejemplos = "; ".join(f"«{t}» ({y})" for t, y in zip(ej["titulo"], ej["anio"]))
            h.append(f"Coautoría real: {self.doc.at[a, 'nombre']} — {self.doc.at[b, 'nombre']}{inter}: {ed['n']} obra(s) "
                     f"conjunta(s) entre {ed['ini']} y {ed['fin']}. Ejemplo: {ejemplos}.")
        for obra, sim in ctx.entidades["obras"][:8]:
            if obra in self.obras.index:
                aut = ", ".join(self.doc.at[d, "nombre"] for d in self.autores.get(obra, []))
                h.append(f"Obra relacionada con la pregunta: «{self.obras.at[obra, 'titulo']}» ({self.obras.at[obra, 'anio']}); "
                         f"docentes ECMC autores: {aut}.")
        for c in caminos:
            if c[1] is None:
                h.append(f"Camino: no existe ninguna cadena de coautorías entre {self._nombre(c[0])} y {self._nombre(c[2])}.")
            else:
                h.append(f"Camino más corto en la red de coautoría ({len(c) - 1} salto/s): "
                         + " → ".join(self.doc.at[x, "nombre"] for x in c) + ".")
        for a, b, p in ctx.aristas_pred:
            h.append(f"Predicción de GraphSAGE (no es una colaboración real): {self._nombre(a)} y {self._nombre(b)} aún no "
                     f"han publicado juntos; probabilidad estimada de colaborar = {p:.2f}.")
        if "comunidad" in ctx.entidades["intenciones"]:
            grupos = {}
            for d, c in self.comunidad_de.items():
                grupos.setdefault(c, []).append(self.doc.at[d, "nombre"])
            for c, miembros in sorted(grupos.items()):
                if c != "Sin coautores":
                    h.append(f"Comunidad {c} (Louvain): {', '.join(sorted(miembros))}.")


# ------------------------------------------------------------ 5. generación con Claude
SISTEMA = """Respondes preguntas sobre la red de coautoría de los docentes de la Escuela de Ciencias Matemáticas y \
Computacionales (ECMC). Usa solo los hechos numerados del contexto; cita cada afirmación con su número entre \
corchetes, por ejemplo [3]. Distingue siempre entre coautorías reales y predicciones de GraphSAGE, que son \
estimaciones y no colaboraciones existentes. Si el contexto no alcanza para responder, dilo y explica qué falta. \
Responde en español, en uno o dos párrafos breves o una lista corta."""


def responder_con_claude(pregunta: str, ctx: Contexto, api_key: str, modelo: str = MODELO_POR_DEFECTO) -> dict:
    """Devuelve {"texto": ..., "error": ...}. Nunca lanza excepciones hacia la app."""
    import anthropic

    cliente = anthropic.Anthropic(api_key=api_key)
    extra = {}
    if modelo in MODELOS_CON_FALLBACK:   # si el modelo declina, el servidor reintenta con el recomendado
        extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
    try:
        r = cliente.beta.messages.create(
            model=modelo,
            max_tokens=16000,
            output_config={"effort": "low"},
            system=SISTEMA,
            messages=[{"role": "user", "content": f"<contexto>\n{ctx.texto()}\n</contexto>\n\nPregunta: {pregunta}"}],
            **extra,
        )
    except anthropic.AuthenticationError:
        return {"texto": None, "error": "La clave ANTHROPIC_API_KEY no es válida."}
    except anthropic.PermissionDeniedError:
        return {"texto": None, "error": "La clave no tiene permiso para usar este modelo."}
    except anthropic.NotFoundError:
        return {"texto": None, "error": f"El modelo «{modelo}» no existe o no está disponible para esta clave."}
    except anthropic.RateLimitError:
        return {"texto": None, "error": "Límite de uso alcanzado o sin créditos. Intenta en un momento."}
    except anthropic.APIStatusError as ex:
        return {"texto": None, "error": f"Error de la API ({ex.status_code}): {ex.message}"}
    except anthropic.APIConnectionError:
        return {"texto": None, "error": "No se pudo conectar con la API de Anthropic."}
    if r.stop_reason == "refusal":
        return {"texto": None, "error": "El modelo declinó responder esta pregunta."}
    texto = "".join(b.text for b in r.content if b.type == "text").strip()
    return {"texto": texto or None, "error": None if texto else "La respuesta llegó vacía.", "modelo": r.model}


# ------------------------------------------------------------ 5b. generación con OpenAI (alternativa)
OPENAI_MODELO_POR_DEFECTO = "gpt-5.4-mini"


def responder_con_openai(pregunta: str, ctx: Contexto, api_key: str, modelo: str = OPENAI_MODELO_POR_DEFECTO) -> dict:
    """Misma interfaz que responder_con_claude, usando la Responses API de OpenAI."""
    import openai

    cliente = openai.OpenAI(api_key=api_key)
    try:
        r = cliente.responses.create(
            model=modelo,
            instructions=SISTEMA,
            input=f"<contexto>\n{ctx.texto()}\n</contexto>\n\nPregunta: {pregunta}",
            max_output_tokens=4000,
        )
    except openai.AuthenticationError:
        return {"texto": None, "error": "La clave OPENAI_API_KEY no es válida."}
    except openai.PermissionDeniedError:
        return {"texto": None, "error": "La clave no tiene permiso para usar este modelo."}
    except openai.NotFoundError:
        return {"texto": None, "error": f"El modelo «{modelo}» no existe para esta clave. Cambia OPENAI_MODEL en Secrets."}
    except openai.RateLimitError:
        return {"texto": None, "error": "Límite de uso alcanzado o sin saldo en la cuenta de OpenAI."}
    except openai.APIStatusError as ex:
        return {"texto": None, "error": f"Error de la API de OpenAI ({ex.status_code}): {ex.message}"}
    except openai.APIConnectionError:
        return {"texto": None, "error": "No se pudo conectar con la API de OpenAI."}
    texto = (r.output_text or "").strip()
    return {"texto": texto or None, "error": None if texto else "La respuesta llegó vacía.", "modelo": r.model}


# ------------------------------------------------------------ 5c. generación con NVIDIA (API compatible con OpenAI)
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODELO_POR_DEFECTO = "meta/llama-3.3-70b-instruct"


def responder_con_nvidia(pregunta: str, ctx: Contexto, api_key: str, modelo: str = NVIDIA_MODELO_POR_DEFECTO) -> dict:
    """NVIDIA build.nvidia.com expone modelos abiertos con la API de Chat Completions de OpenAI."""
    import openai

    cliente = openai.OpenAI(api_key=api_key, base_url=NVIDIA_BASE_URL)
    try:
        r = cliente.chat.completions.create(
            model=modelo,
            messages=[{"role": "system", "content": SISTEMA},
                      {"role": "user", "content": f"<contexto>\n{ctx.texto()}\n</contexto>\n\nPregunta: {pregunta}"}],
            max_tokens=2048,
            temperature=0.2,
        )
    except openai.AuthenticationError:
        return {"texto": None, "error": "La clave NVIDIA_API_KEY no es válida (debe empezar con «nvapi-»)."}
    except openai.PermissionDeniedError:
        return {"texto": None, "error": "La clave no tiene permiso para usar este modelo de NVIDIA."}
    except openai.NotFoundError:
        return {"texto": None, "error": f"El modelo «{modelo}» no está disponible en NVIDIA. Cambia NVIDIA_MODEL en Secrets."}
    except openai.RateLimitError:
        return {"texto": None, "error": "Límite de solicitudes o de créditos de NVIDIA alcanzado. Espera un minuto."}
    except openai.APIStatusError as ex:
        return {"texto": None, "error": f"Error de la API de NVIDIA ({ex.status_code}): {ex.message}"}
    except openai.APIConnectionError:
        return {"texto": None, "error": "No se pudo conectar con la API de NVIDIA."}
    texto = (r.choices[0].message.content or "").strip() if r.choices else ""
    return {"texto": texto or None, "error": None if texto else "La respuesta llegó vacía.", "modelo": r.model}
