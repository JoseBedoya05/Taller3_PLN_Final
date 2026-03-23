import os
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Taller 3 - Demo RAG vs Fine-Tuning", layout="wide")

st.title("Taller 3 - Demo de resultados")
st.caption("Demo ligera para visualizar el corpus preparado y los resultados de la comparación.")

DATA_DIR = Path("data")
chunks_path = DATA_DIR / "chunks.parquet"
results_path = DATA_DIR / "comparacion_detallada.csv"
summary_path = DATA_DIR / "comparacion_por_tarea.csv"

missing = [str(p) for p in [chunks_path, results_path, summary_path] if not p.exists()]
if missing:
    st.warning(
        "Faltan archivos en la carpeta data/.\n\n"
        "Debes copiar desde Colab al menos:\n"
        "- chunks.parquet\n"
        "- comparacion_detallada.csv\n"
        "- comparacion_por_tarea.csv\n\n"
        f"Archivos faltantes: {missing}"
    )
    st.stop()

chunks_df = pd.read_parquet(chunks_path)
results_df = pd.read_csv(results_path)
summary_df = pd.read_csv(summary_path)

st.subheader("Resumen de resultados")
st.dataframe(summary_df, use_container_width=True)

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Filtrar ejemplos")
    task_options = ["Todos"] + sorted(results_df["task_type"].dropna().astype(str).unique().tolist())
    selected_task = st.selectbox("Tipo de tarea", task_options)

    query = st.text_input("Buscar palabra o frase", "")

with col2:
    filtered = results_df.copy()

    if selected_task != "Todos":
        filtered = filtered[filtered["task_type"].astype(str) == selected_task]

    if query.strip():
        mask = (
            filtered["question"].astype(str).str.contains(query, case=False, na=False)
            | filtered["reference_answer"].astype(str).str.contains(query, case=False, na=False)
            | filtered["ft_prediction"].astype(str).str.contains(query, case=False, na=False)
            | filtered["rag_prediction"].astype(str).str.contains(query, case=False, na=False)
        )
        filtered = filtered[mask]

    st.subheader("Comparación detallada")
    st.dataframe(
        filtered[
            [
                "example_id", "task_type", "question", "reference_answer",
                "ft_prediction", "rag_prediction",
                "ft_primary_metric", "rag_primary_metric",
                "retrieved_files"
            ]
        ],
        use_container_width=True,
        height=420
    )

st.subheader("Explorador del corpus")
file_options = ["Todos"] + sorted(chunks_df["file_name"].dropna().astype(str).unique().tolist())
selected_file = st.selectbox("Archivo", file_options)

corpus_view = chunks_df.copy()
if selected_file != "Todos":
    corpus_view = corpus_view[corpus_view["file_name"].astype(str) == selected_file]

chunk_search = st.text_input("Buscar dentro del corpus", "", key="chunk_search")
if chunk_search.strip():
    corpus_view = corpus_view[corpus_view["text"].astype(str).str.contains(chunk_search, case=False, na=False)]

st.dataframe(corpus_view[["chunk_id", "file_name", "text"]], use_container_width=True, height=360)

st.info(
    "Esta app es una demo ligera para mostrar el corpus y los resultados. "
    "El entrenamiento y la evaluación completa se hacen en Google Colab."
)
