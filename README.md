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
4. **Generación.** Un LLM responde citando los hechos con [n] y distingue las colaboraciones reales de las predichas. Puede ser GPT (`gpt-5.4-mini` por defecto) o Claude (`claude-opus-5-5` por defecto).

**Clave de API:** en Streamlit Community Cloud ve a *Manage app → Settings → Secrets* y agrega:

```toml
OPENAI_API_KEY = "sk-..."
# OPENAI_MODEL = "gpt-5.4-mini"      # opcional
# o bien, para usar Claude:
# ANTHROPIC_API_KEY = "sk-ant-..."
# CLAUDE_MODEL = "claude-opus-5-5"   # opcional
```

Si están ambas claves, se usa la de OpenAI.

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
