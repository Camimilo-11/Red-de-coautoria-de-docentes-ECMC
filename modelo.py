"""GraphSAGE (agregador de media) para predicción de coautorías entre docentes.

Implementado desde cero sobre NumPy para que la app no dependa de PyTorch.
Las funciones aceptan el módulo numérico `xp` (numpy o autograd.numpy), así
el mismo forward sirve para entrenar (train.py) y para inferir (app.py).
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
DEPARTAMENTOS = ["Matemáticas", "Computación", "Desarrollo Humano"]


# ---------------------------------------------------------------- datos
def cargar_datos(carpeta: Path = DATA) -> dict:
    docentes = pd.read_csv(carpeta / "docentes.csv")
    part = pd.read_csv(carpeta / "participaciones.csv")
    obra_sub = pd.read_csv(carpeta / "obra_sublinea.csv")
    lineas = pd.read_csv(carpeta / "lineas.csv")
    return {"docentes": docentes, "part": part, "obra_sub": obra_sub, "lineas": lineas}


def coautorias(datos: dict) -> pd.DataFrame:
    """Un registro por par de docentes: año de la 1.ª colaboración, última y n.º de obras."""
    part = datos["part"]
    filas = []
    for obra, g in part.groupby("id_obra"):
        ids = sorted(set(g["id_docente"]))
        anio = int(g["anio"].iloc[0])
        for a, b in combinations(ids, 2):
            filas.append((a, b, obra, anio))
    df = pd.DataFrame(filas, columns=["a", "b", "obra", "anio"])
    return (df.groupby(["a", "b"])
              .agg(primer_anio=("anio", "min"), ultimo_anio=("anio", "max"), n_obras=("obra", "nunique"))
              .reset_index())


def aristas_hasta(coaut: pd.DataFrame, nodos: list, corte: int) -> np.ndarray:
    idx = {n: i for i, n in enumerate(nodos)}
    sub = coaut[coaut["primer_anio"] <= corte]
    return np.array([(idx[a], idx[b]) for a, b in zip(sub["a"], sub["b"])], dtype=int).reshape(-1, 2)


def matriz_media(aristas: np.ndarray, n: int) -> np.ndarray:
    """Adyacencia normalizada por filas (D^-1 A). Nodos sin vecinos quedan en cero."""
    A = np.zeros((n, n))
    if len(aristas):
        A[aristas[:, 0], aristas[:, 1]] = 1.0
        A[aristas[:, 1], aristas[:, 0]] = 1.0
    grado = A.sum(1, keepdims=True)
    return np.divide(A, grado, out=np.zeros_like(A), where=grado > 0)


def perfil_tematico(datos: dict, corte: int) -> pd.DataFrame:
    """Proporción de obras de cada docente por sublínea, usando solo obras con año <= corte."""
    part = datos["part"][datos["part"]["anio"] <= corte]
    m = part.merge(datos["obra_sub"], on="id_obra")
    tabla = pd.crosstab(m["id_docente"], m["id_sublinea"])
    tabla = tabla.reindex(index=datos["docentes"]["id"], columns=datos["lineas"]["id_sublinea"], fill_value=0)
    suma = tabla.sum(axis=1).replace(0, 1)
    return tabla.div(suma, axis=0)


def atributos(datos: dict, corte: int, stats_pub: tuple | None = None):
    """Matriz X: departamento (one-hot) + log(n.º obras) estandarizado + perfil por sublíneas.

    Todo se calcula con información hasta `corte` para evitar fuga temporal.
    """
    doc = datos["docentes"]
    dep = np.stack([(doc["departamento"] == d).to_numpy(float) for d in DEPARTAMENTOS], 1)
    n_pub = (datos["part"][datos["part"]["anio"] <= corte]
             .groupby("id_docente")["id_obra"].nunique()
             .reindex(doc["id"], fill_value=0).to_numpy(float))
    lp = np.log1p(n_pub)
    if stats_pub is None:
        stats_pub = (float(lp.mean()), float(lp.std() or 1.0))
    lp = (lp - stats_pub[0]) / stats_pub[1]
    perfil = perfil_tematico(datos, corte).to_numpy(float)
    X = np.hstack([dep, lp[:, None], perfil])
    return X, stats_pub


# ---------------------------------------------------------------- modelo
def iniciar_parametros(rng, d_in, ocultas=32, salida=16, decod=16):
    def glorot(i, o):
        return rng.normal(0, np.sqrt(2.0 / (i + o)), (i, o))
    return {
        "W1s": glorot(d_in, ocultas), "W1n": glorot(d_in, ocultas), "b1": np.zeros(ocultas),
        "W2s": glorot(ocultas, salida), "W2n": glorot(ocultas, salida), "b2": np.zeros(salida),
        "Wd": glorot(2 * salida, decod), "bd": np.zeros(decod),
        "wo": glorot(decod, 1)[:, 0], "bo": 0.0,
    }


def embeddings(p, X, A, xp=np, mascara=None):
    """Dos capas GraphSAGE-media: h = ReLU(X·Ws + mean(vecinos)·Wn + b)."""
    h = xp.maximum(0, xp.dot(X, p["W1s"]) + xp.dot(xp.dot(A, X), p["W1n"]) + p["b1"])
    if mascara is not None:          # dropout en entrenamiento
        h = h * mascara
    z = xp.dot(h, p["W2s"]) + xp.dot(xp.dot(A, h), p["W2n"]) + p["b2"]
    return z / (xp.sqrt(xp.sum(z ** 2, axis=1, keepdims=True)) + 1e-9)


def logits_pares(p, Z, pares, xp=np):
    """Decodificador simétrico: MLP sobre [z_u ⊙ z_v, |z_u − z_v|]."""
    zu, zv = Z[pares[:, 0]], Z[pares[:, 1]]
    f = xp.concatenate([zu * zv, xp.abs(zu - zv)], axis=1)
    h = xp.maximum(0, xp.dot(f, p["Wd"]) + p["bd"])
    return xp.dot(h, p["wo"]) + p["bo"]


def sigmoide(x):
    return 1.0 / (1.0 + np.exp(-x))


def predecir(lista_params, X, A, pares):
    """Probabilidad media del ensamble de modelos (uno por semilla)."""
    return np.mean([sigmoide(logits_pares(p, embeddings(p, X, A), pares)) for p in lista_params], 0)


# ---------------------------------------------------------------- líneas base
def heuristicas(aristas: np.ndarray, n: int, pares: np.ndarray, X_perfil: np.ndarray) -> dict:
    A = np.zeros((n, n))
    if len(aristas):
        A[aristas[:, 0], aristas[:, 1]] = A[aristas[:, 1], aristas[:, 0]] = 1
    grado = A.sum(1)
    u, v = pares[:, 0], pares[:, 1]
    cn = (A[u] * A[v]).sum(1)
    inv_log = np.where(grado > 1, 1 / np.log(np.maximum(grado, 2)), 0)
    aa = (A[u] * A[v] * inv_log).sum(1)
    union = ((A[u] + A[v]) > 0).sum(1)
    jac = np.divide(cn, union, out=np.zeros_like(cn), where=union > 0)
    pa = grado[u] * grado[v]
    nu = np.linalg.norm(X_perfil, axis=1) + 1e-9
    cos = (X_perfil[u] * X_perfil[v]).sum(1) / (nu[u] * nu[v])
    return {"Vecinos comunes": cn, "Adamic-Adar": aa, "Jaccard": jac,
            "Apego preferencial": pa, "Similitud temática (coseno)": cos}
