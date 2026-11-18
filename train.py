"""Entrena GraphSAGE, lo evalúa con división temporal y guarda model.pkl.

Uso:
    python train.py            # corte de evaluación 2023, 10 semillas

Protocolo
---------
1. Evaluación: se entrena con las coautorías iniciadas hasta CORTE y con
   atributos calculados solo con obras hasta CORTE. Se evalúa sobre todos los
   pares aún no conectados en CORTE: positivos = pares que empiezan a
   colaborar después de CORTE.
2. Modelo final: se re-entrena con toda la red (hasta el último año) para
   generar las recomendaciones que muestra la app.
Los hiperparámetros se fijaron a priori (no se ajustaron mirando el test).
"""
import pickle
from datetime import date

import autograd.numpy as anp
import numpy as np
import pandas as pd
from autograd import grad
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

import modelo as M

CORTE = 2023
SEMILLAS = list(range(10))
HP = dict(ocultas=32, salida=16, decod=16, epocas=300, lr=0.01, wd=1e-4,
          p_drop=0.3, frac_supervision=0.3, negativos_por_positivo=3)


def entrenar(X, aristas, n, semilla):
    rng = np.random.default_rng(semilla)
    p = M.iniciar_parametros(rng, X.shape[1], HP["ocultas"], HP["salida"], HP["decod"])
    existentes = {tuple(sorted(e)) for e in aristas.tolist()}
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v_ = {k: np.zeros_like(v) for k, v in p.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8

    for t in range(1, HP["epocas"] + 1):
        # Separar aristas de mensaje y de supervisión: el modelo no "ve" la arista que debe predecir
        perm = rng.permutation(len(aristas))
        k = max(1, int(HP["frac_supervision"] * len(aristas)))
        sup, msg = aristas[perm[:k]], aristas[perm[k:]]
        A = M.matriz_media(msg, n)
        neg = []
        while len(neg) < k * HP["negativos_por_positivo"]:
            u, w = rng.integers(0, n, 2)
            if u != w and (min(u, w), max(u, w)) not in existentes:
                neg.append((u, w))
        pares = np.vstack([sup, np.array(neg)])
        y = np.r_[np.ones(len(sup)), np.zeros(len(neg))]
        mascara = (rng.random((n, HP["ocultas"])) > HP["p_drop"]) / (1 - HP["p_drop"])

        def perdida(params):
            z = M.embeddings(params, X, A, xp=anp, mascara=mascara)
            lg = M.logits_pares(params, z, pares, xp=anp)
            bce = anp.mean(anp.logaddexp(0, lg) - y * lg)
            l2 = sum(anp.sum(params[c] ** 2) for c in params if c.startswith("W"))
            return bce + HP["wd"] * l2

        g = grad(perdida)(p)
        for c in p:  # Adam
            m[c] = b1 * m[c] + (1 - b1) * g[c]
            v_[c] = b2 * v_[c] + (1 - b2) * g[c] ** 2
            p[c] = p[c] - HP["lr"] * (m[c] / (1 - b1 ** t)) / (np.sqrt(v_[c] / (1 - b2 ** t)) + eps)
    return {c: np.asarray(val, dtype=float) for c, val in p.items()}


def curva(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    return {"fpr": fpr, "tpr": tpr, "auc": float(roc_auc_score(y, s)),
            "ap": float(average_precision_score(y, s))}


def main():
    datos = M.cargar_datos()
    nodos = datos["docentes"]["id"].tolist()
    n = len(nodos)
    coaut = M.coautorias(datos)

    # ---------------- evaluación temporal
    tr = M.aristas_hasta(coaut, nodos, CORTE)
    X, stats = M.atributos(datos, CORTE)
    A = M.matriz_media(tr, n)
    conectados = {tuple(sorted(e)) for e in tr.tolist()}
    cand = np.array([(i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in conectados])
    todas = {tuple(sorted(e)) for e in M.aristas_hasta(coaut, nodos, 9999).tolist()}
    y = np.array([tuple(c) in todas for c in cand.tolist()], dtype=int)
    grado = np.bincount(tr.ravel(), minlength=n)
    frio = (grado[cand[:, 0]] == 0) | (grado[cand[:, 1]] == 0)

    params_eval, por_semilla = [], []
    for s in SEMILLAS:
        p = entrenar(X, tr, n, s)
        params_eval.append(p)
        sc = M.sigmoide(M.logits_pares(p, M.embeddings(p, X, A), cand))
        por_semilla.append({"semilla": s,
                            "auc": roc_auc_score(y, sc), "ap": average_precision_score(y, sc),
                            "auc_frio": roc_auc_score(y[frio], sc[frio]),
                            "ap_frio": average_precision_score(y[frio], sc[frio])})
        print(f"semilla {s}: AUC={por_semilla[-1]['auc']:.3f}  AUC frío={por_semilla[-1]['auc_frio']:.3f}")

    puntajes = {"GraphSAGE (ensamble)": M.predecir(params_eval, X, A, cand)}
    puntajes.update(M.heuristicas(tr, n, cand, X[:, 4:]))
    resultados = {"global": {}, "frio": {}}
    for nombre, sc in puntajes.items():
        resultados["global"][nombre] = curva(y, sc)
        resultados["frio"][nombre] = curva(y[frio], sc[frio])
    # Bootstrap pareado (1000 remuestreos de pares candidatos): IC 95 % del AUC y de la diferencia vs. GraphSAGE
    rng = np.random.default_rng(0)
    gs = puntajes["GraphSAGE (ensamble)"]
    for clave, mask in (("global", np.ones(len(cand), bool)), ("frio", frio)):
        idx_all = np.flatnonzero(mask)
        muestras = {k: [] for k in puntajes}
        difs = {k: [] for k in puntajes}
        for _ in range(1000):
            idx = rng.choice(idx_all, len(idx_all))
            if y[idx].min() == y[idx].max():
                continue
            a_gs = roc_auc_score(y[idx], gs[idx])
            for k, sc in puntajes.items():
                a = roc_auc_score(y[idx], sc[idx])
                muestras[k].append(a)
                difs[k].append(a_gs - a)
        for k in puntajes:
            resultados[clave][k]["ic95"] = [float(np.percentile(muestras[k], 2.5)), float(np.percentile(muestras[k], 97.5))]
            resultados[clave][k]["dif_ic95"] = [float(np.percentile(difs[k], 2.5)), float(np.percentile(difs[k], 97.5))]

    for nombre in puntajes:
        print(f"{nombre:32s} AUC={resultados['global'][nombre]['auc']:.3f} "
              f"AP={resultados['global'][nombre]['ap']:.3f} | frío AUC={resultados['frio'][nombre]['auc']:.3f}")

    # ---------------- ¿Las colaboraciones existentes se repiten? (evaluación, mismo protocolo temporal)
    # Pares que ya colaboraron hasta CORTE; positivo = vuelven a publicar juntos en los 3 años siguientes.
    from sklearn.linear_model import LogisticRegression
    pp = M.pares_por_obra(datos)
    H = M.HORIZONTE_REP
    entreno = pd.concat([M.variables_repeticion(datos, pp, t) for t in range(2010, CORTE - H + 1)], ignore_index=True)
    prueba = M.variables_repeticion(datos, pp, CORTE)
    V = M.VARIABLES_REP
    Xe = entreno[V].to_numpy(float)
    mu, sd = Xe.mean(0), Xe.std(0)
    sd[sd == 0] = 1
    lr = LogisticRegression(C=1.0, max_iter=1000).fit((Xe - mu) / sd, entreno["repite"])
    y_rep = prueba["repite"].to_numpy()
    idx = {d: i for i, d in enumerate(nodos)}
    afinidad = []
    for a, b in zip(prueba["a"], prueba["b"]):   # GraphSAGE con la arista oculta: «¿la esperaría si no la viera?»
        ia, ib = idx[a], idx[b]
        sin = tr[[{x, w} != {ia, ib} for x, w in tr.tolist()]]
        afinidad.append(float(M.predecir(params_eval, X, M.matriz_media(sin, n), np.array([[ia, ib]]))[0]))
    puntajes_rep = {
        "GraphSAGE (afinidad, arista oculta)": np.array(afinidad),
        "Regresión logística (7 variables)": lr.predict_proba((prueba[V].to_numpy(float) - mu) / sd)[:, 1],
        "Recencia (−años desde la última)": -prueba["anios_desde_ultima"].to_numpy(float),
        "N.º de obras conjuntas": prueba["n"].to_numpy(float),
        "Colaboró en los últimos 3 años": prueba["conjuntas_ultimos_3"].to_numpy(float),
    }
    curvas_rep = {k: curva(y_rep, v) for k, v in puntajes_rep.items()}
    rng = np.random.default_rng(1)
    for k, v in puntajes_rep.items():
        bs = []
        for _ in range(1000):
            i = rng.integers(0, len(y_rep), len(y_rep))
            if y_rep[i].min() != y_rep[i].max():
                bs.append(roc_auc_score(y_rep[i], v[i]))
        curvas_rep[k]["ic95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        print(f"[repetición] {k:38s} AUC={curvas_rep[k]['auc']:.3f} IC95=[{curvas_rep[k]['ic95'][0]:.2f}, {curvas_rep[k]['ic95'][1]:.2f}]")

    # ---------------- modelo final con toda la red
    corte_final = int(datos["part"]["anio"].max())
    tr_f = M.aristas_hasta(coaut, nodos, corte_final)
    X_f, stats_f = M.atributos(datos, corte_final)
    params_final = [entrenar(X_f, tr_f, n, s) for s in SEMILLAS]

    paquete = {
        "descripcion": "GraphSAGE (media) + decodificador MLP, ensamble de 10 semillas, NumPy puro",
        "fecha_entrenamiento": date.today().isoformat(),
        "hiperparametros": HP,
        "nodos": nodos,
        "departamentos": M.DEPARTAMENTOS,
        "corte_evaluacion": CORTE,
        "corte_final": corte_final,
        "stats_pub_final": stats_f,
        "params_final": params_final,
        "repeticion": {"corte": CORTE, "horizonte": H, "n_entreno": int(len(entreno)),
                       "pos_entreno": int(entreno["repite"].sum()), "n_prueba": int(len(prueba)),
                       "pos_prueba": int(y_rep.sum()), "curvas": curvas_rep},
        "evaluacion": {
            "n_candidatos": int(len(cand)), "n_positivos": int(y.sum()),
            "n_candidatos_frio": int(frio.sum()), "n_positivos_frio": int(y[frio].sum()),
            "por_semilla": por_semilla, "curvas": resultados,
        },
    }
    with open("model.pkl", "wb") as f:
        pickle.dump(paquete, f)
    print("Guardado model.pkl")


if __name__ == "__main__":
    main()
