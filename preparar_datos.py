"""Extrae del Excel consolidado los CSV mínimos que usa la app.

Uso:
    python preparar_datos.py "ECMC_Consolidado_Historico_Corregido.xlsx"

Solo se conserva la producción FORMAL (columna «Formal (1/0)» = 1), como
define la hoja Guia_y_alcance del consolidado.
"""
import sys
from pathlib import Path

import pandas as pd

SALIDA = Path(__file__).parent / "data"


def main(ruta_excel: str) -> None:
    xl = pd.ExcelFile(ruta_excel)
    SALIDA.mkdir(exist_ok=True)

    docentes = xl.parse("Docentes")[["ID docente", "Nombre de presentación", "Departamento"]]
    docentes.columns = ["id", "nombre", "departamento"]
    docentes.to_csv(SALIDA / "docentes.csv", index=False)

    dp = xl.parse("Docente_Publicacion")
    dp = dp[dp["Formal (1/0)"] == 1][["ID docente", "ID obra", "Año de análisis"]]
    dp.columns = ["id_docente", "id_obra", "anio"]
    dp = dp.dropna().drop_duplicates()
    dp["anio"] = dp["anio"].astype(int)
    dp.to_csv(SALIDA / "participaciones.csv", index=False)

    ol = xl.parse("Obra_Linea_Sublinea")[["ID obra", "ID línea", "ID sublínea"]]
    ol.columns = ["id_obra", "id_linea", "id_sublinea"]
    ol = ol[ol["id_obra"].isin(dp["id_obra"])].dropna().drop_duplicates()
    ol.to_csv(SALIDA / "obra_sublinea.csv", index=False)

    ls = xl.parse("Lineas_y_Sublineas")
    ls.columns = ["id_linea", "linea", "id_sublinea", "sublinea"]
    ls.to_csv(SALIDA / "lineas.csv", index=False)

    # Control: las coautorías derivadas deben coincidir con la hoja oficial
    oficial = xl.parse("Coautorias_Escuela")
    pares_oficiales = {tuple(sorted(p)) for p in zip(oficial["ID docente A"], oficial["ID docente B"])}
    por_obra = dp.groupby("id_obra")["id_docente"].apply(lambda s: sorted(set(s)))
    pares = {(a, b) for ds in por_obra for i, a in enumerate(ds) for b in ds[i + 1:]}
    print(f"Docentes: {len(docentes)} · participaciones formales: {len(dp)} · obras: {dp.id_obra.nunique()}")
    print(f"Pares derivados: {len(pares)} · pares hoja oficial: {len(pares_oficiales)} · "
          f"coinciden: {pares == pares_oficiales}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
