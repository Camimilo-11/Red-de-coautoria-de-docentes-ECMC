# Red de coautoría de docentes ECMC · GraphSAGE

App en Streamlit que visualiza la red de coautoría de los 50 docentes de la ECMC
(Matemáticas, Computación y Desarrollo Humano) y usa **GraphSAGE** para predecir
qué pares de docentes probablemente colaborarán.

## Qué muestra la app

| Pestaña | Contenido |
|---|---|
| 🕸️ Red | Grafo interactivo con colores por departamento, filtro por año y colaboraciones predichas (en punteado) |
| 🧭 Perfil temático | Mapa de calor docente × línea de investigación |
| 🗺️ Mapa de embeddings | Proyección t-SNE de los embeddings que aprende GraphSAGE |
| 🤝 Recomendador | Top 5 colaboradores sugeridos para cada docente, con sublíneas en común y el camino que los une en la red |
| 📈 Evaluación | Curvas ROC, AUC con IC 95 % y comparación con líneas base |

## Modelo

- **GraphSAGE de 2 capas** con agregador de media y un decodificador MLP simétrico. Es un ensamble de 10 semillas.
- Está implementado **desde cero en NumPy** (`modelo.py`), así que la app no depende de PyTorch y se despliega liviana.
- **Atributos de cada docente:** departamento, log del número de obras y proporción de obras en cada una de las 46 sublíneas.
- **Evaluación temporal:** se entrena con la red hasta 2023 y se predicen las colaboraciones nuevas de 2024–2026 (33 pares).

| Método | AUC-ROC (todos los pares) | AUC-ROC (docentes sin historial) |
|---|---|---|
| **GraphSAGE (ensamble)** | **0.841** [0.778, 0.894] | **0.769** [0.665, 0.871] |
| Apego preferencial | 0.729 | 0.500 |
| Similitud temática (coseno) | 0.725 | 0.548 |
| Adamic-Adar | 0.627 | 0.500 |
| Vecinos comunes | 0.626 | 0.500 |

**Limitaciones:**

- La red es pequeña (33 positivos de prueba, 12 de ellos en el subconjunto sin historial).
- Los perfiles temáticos se infieren de los títulos.
- 7 docentes no tienen producción formal, así que se excluyen de las recomendaciones.

## Estructura del repositorio

```
app.py               ← la app de Streamlit
modelo.py            ← GraphSAGE (forward en NumPy), atributos y líneas base
model.pkl            ← pesos entrenados + resultados de evaluación
train.py             ← re-entrena y regenera model.pkl
preparar_datos.py    ← genera data/*.csv a partir del Excel consolidado
requirements.txt
data/                ← CSV derivados que lee la app
.streamlit/config.toml
```

## Ejecutar en local

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Para re-entrenar el modelo (tarda unos 20 s):

```bash
python preparar_datos.py "ECMC_Consolidado_Historico_Corregido.xlsx"   # solo si cambió el Excel
python train.py
```

## Fuente de datos

Consolidado histórico corregido de la ECMC (corte 29/10/2026). Solo se usa la producción formal.
El Excel original no se versiona; `data/` contiene solo las tablas mínimas derivadas.
