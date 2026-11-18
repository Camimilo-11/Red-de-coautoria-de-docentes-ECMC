# Red de coautoría de docentes ECMC · GraphSAGE

App en Streamlit que visualiza la red de coautoría de los 50 docentes de la ECMC
(Matemáticas, Computación y Desarrollo Humano) y usa **GraphSAGE** para predecir
qué pares de docentes probablemente colaborarán.

## Qué muestra la app

| Pestaña | Contenido |
|---|---|
| 🕸️ Red | Grafo interactivo con colores por departamento, filtro por año y colaboraciones predichas (en punteado) |
| 💬 Pregúntale a la red | **Graph RAG**: pregunta en lenguaje natural → recuperación en el grafo → respuesta de un LLM (GPT o Claude) con citas |
| 🧭 Perfil temático | Mapa de calor docente × línea de investigación |
| 🗺️ Mapa de embeddings | Proyección t-SNE de los embeddings que aprende GraphSAGE |
| 🤝 Recomendador | Filtro **Nuevos / Existentes / Todos**: afinidad GraphSAGE con docentes con los que aún no publica y con sus coautores actuales (calculada ocultando la colaboración), estado activa/inactiva, sublíneas en común y camino en la red |
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

**Colaboraciones existentes.** También se evaluó si las colaboraciones existentes hasta 2023 se repiten en 2024–2026 (40 pares, 7 repitieron). Ni la afinidad de GraphSAGE (AUC 0.41) ni una regresión logística con 7 variables (AUC 0.45) lo anticiparon. La señal más útil fue la recencia (AUC 0.74, IC 95 % 0.58–0.86). Por eso la app muestra la afinidad y el estado de la colaboración, pero no una "probabilidad de repetir".

**Limitaciones:**

- La red es pequeña (33 positivos de prueba, 12 de ellos en el subconjunto sin historial).
- Los perfiles temáticos se infieren de los títulos.
- 7 docentes no tienen producción formal, así que se excluyen de las recomendaciones.

## Estructura del repositorio

```
app.py               ← la app de Streamlit
graphrag.py          ← Graph RAG: enlace de entidades, recuperación en el grafo y llamada al LLM
modelo.py            ← GraphSAGE (forward en NumPy), atributos y líneas base
model.pkl            ← pesos entrenados + resultados de evaluación
train.py             ← re-entrena y regenera model.pkl
preparar_datos.py    ← genera data/*.csv a partir del Excel consolidado
requirements.txt
data/                ← CSV derivados que lee la app
.streamlit/config.toml
```

## Graph RAG

1. **Enlace de entidades.** Se detectan docentes nombrados, departamentos y sublíneas (diccionario de términos en español e inglés), y se buscan obras por similitud TF-IDF de n-gramas de caracteres sobre los títulos.
2. **Recuperación en el grafo.** Se eligen los docentes semilla, se expanden a sus coautores (1 salto) y se calculan los caminos más cortos entre los docentes nombrados (multi-salto). Si se pregunta por el futuro, se agregan las colaboraciones que predice GraphSAGE.
3. **Contexto.** Se arma una lista de hechos numerados: perfiles, coautorías con un título de ejemplo, caminos y predicciones.
4. **Generación.** Un LLM responde citando los hechos con [n] y distingue las colaboraciones reales de las predichas. Puede ser Gemini (`gemini-3.8-flash` por defecto, plan gratuito), un modelo abierto vía NVIDIA (Nemotron 3 por defecto, con respaldo automático si NVIDIA lo retira), GPT (`gpt-5.4-mini`) o Claude (`claude-opus-5-5`).

**Clave de API:** en Streamlit Community Cloud ve a *Manage app → Settings → Secrets* y agrega:

```toml
GEMINI_API_KEY = "..."               # aistudio.google.com (plan gratuito) — recomendado
# GEMINI_MODEL = "gemini-3.8-flash"  # opcional; si no está disponible, se prueba otro
# o bien:
NVIDIA_API_KEY = "nvapi-..."         # build.nvidia.com (créditos gratuitos)
# NVIDIA_MODEL = "nvidia/nemotron-3-super-120b-a12b"   # opcional; si está retirado, se prueba otro
# o bien:
OPENAI_API_KEY = "sk-..."
# OPENAI_MODEL = "gpt-5.4-mini"      # opcional
# o bien, para usar Claude:
# ANTHROPIC_API_KEY = "sk-ant-..."
# CLAUDE_MODEL = "claude-opus-5-5"   # opcional
```

Usa solo una clave. Si hay varias, la prioridad es Gemini > NVIDIA > OpenAI > Anthropic.

Sin clave, la pestaña funciona en modo extractivo: muestra el subgrafo y los hechos recuperados, sin redactar la respuesta. **Nunca subas la clave al repo**; `.streamlit/secrets.toml` está en `.gitignore`.

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
