import os
import time
import requests
import pandas as pd

# ============================================================
# CONFIG ESCRUTINIOS
# ============================================================

BASE_API = "https://api.eduardolosilla.es"

# Endpoint histórico REAL (confirmado)
ENDPOINT_ESCRUTINIOS = (
    "/escrutinios"
    "?num_jornada={jornada}"
    "&num_temporada={temporada}"
    "&uts={uts}"
)

# Temporada numérica
NUM_TEMPORADA = 2026

# UTS válido
UTS_VALIDO = 1770053592050

# Token de autenticación (definir en variable de entorno)
AUTH_TOKEN = os.getenv("AUTH_TOKEN")

HEADERS_ESCRUTINIOS = {
    "accept": "application/json",
    "authorization": AUTH_TOKEN,
    "user-agent": "Mozilla/5.0",
}

# Jornadas a descargar
JORNADA_INICIAL = 1
JORNADA_FINAL = 44  # Ve cambiando a lo largo de la temporada

# Ritmo seguro
SLEEP_SECONDS_ESCRUTINIOS = 1.25

# Reintentos por robustez
MAX_RETRIES = 3
RETRY_SLEEP = 2.0


# ============================================================
# FUNCIONES ESCRUTINIOS
# ============================================================

def fetch_escrutinios(
    session: requests.Session,
    jornada: int,
    temporada: int,
    uts: int
) -> dict:
    """
    Descarga el JSON de escrutinios para una jornada concreta.
    """
    url = BASE_API + ENDPOINT_ESCRUTINIOS.format(
        jornada=jornada,
        temporada=temporada,
        uts=uts
    )

    last_err = None
    for _ in range(1, MAX_RETRIES + 1):
        r = session.get(url, headers=HEADERS_ESCRUTINIOS, timeout=30)
        if r.status_code == 200:
            return r.json()

        last_err = f"status={r.status_code}, body_head={r.text[:120]}"
        time.sleep(RETRY_SLEEP)

    raise RuntimeError(
        f"Fallo escrutinios jornada {jornada} tras {MAX_RETRIES} intentos: {last_err}"
    )


def normalizar_partidos(payload: dict, jornada: int, temporada: int) -> pd.DataFrame:
    """
    Convierte payload['partidos'] en DataFrame + añade metadatos.
    """
    if "partidos" not in payload or not isinstance(payload["partidos"], list):
        raise ValueError(
            f"Payload inesperado en jornada {jornada}: no existe 'partidos'"
        )

    df = pd.DataFrame(payload["partidos"]).copy()

    # Metadatos para trazabilidad
    df["jornada_consultada"] = jornada
    df["num_temporada"] = temporada

    # Metadatos del propio escrutinio (si existen)
    df["escrutinio"] = payload.get("escrutinio", None)
    df["caducidad"] = payload.get("caducidad", None)

    return df


def descargar_historico(j_ini: int, j_fin: int, temporada: int, uts: int) -> pd.DataFrame:
    """
    Descarga el histórico de escrutinios entre dos jornadas.
    """
    session = requests.Session()

    frames = []
    for jornada in range(j_ini, j_fin + 1):
        print(f"[INFO] Descargando jornada {jornada}")

        payload = fetch_escrutinios(
            session,
            jornada=jornada,
            temporada=temporada,
            uts=uts
        )
        df_j = normalizar_partidos(payload, jornada=jornada, temporada=temporada)

        frames.append(df_j)
        time.sleep(SLEEP_SECONDS_ESCRUTINIOS)

    df_all = pd.concat(frames, ignore_index=True)
    return df_all


def auditoria_basica(df: pd.DataFrame, j_ini: int, j_fin: int) -> dict:
    """
    Busca duplicados/faltantes y hace checks de consistencia.
    """
    out = {}

    # 1) Filas esperadas: 15 partidos por jornada (en quiniela)
    jornadas = list(range(j_ini, j_fin + 1))
    expected_rows = len(jornadas) * 15
    out["rows_total"] = len(df)
    out["expected_rows"] = expected_rows
    out["rows_ok"] = (len(df) == expected_rows)

    # 2) Conteo por jornada
    counts = df.groupby("jornada_consultada").size().reindex(jornadas, fill_value=0)
    out["counts_by_jornada"] = counts

    # 3) Jornadas faltantes / con menos de 15
    out["jornadas_faltantes"] = counts[counts == 0].index.tolist()
    out["jornadas_incompletas"] = counts[(counts > 0) & (counts != 15)].index.tolist()

    # 4) Duplicados: (jornada_consultada, num) debería ser único
    key_cols = ["jornada_consultada"]
    if "num" in df.columns:
        key_cols.append("num")
        dup_mask = df.duplicated(subset=key_cols, keep=False)
        out["duplicados_por_jornada_num"] = int(dup_mask.sum())
    else:
        out["duplicados_por_jornada_num"] = None

    # 5) Orden esperado num=1..15 si existe la columna
    if "num" in df.columns:
        wrong_nums = df[~df["num"].between(1, 15, inclusive="both")]
        out["filas_num_fuera_rango"] = len(wrong_nums)
    else:
        out["filas_num_fuera_rango"] = None

    return out


def run_escrutinios_scraper() -> pd.DataFrame:
    """
    Ejecuta el scraper de escrutinios, audita y guarda el CSV.
    """
    if not AUTH_TOKEN:
        raise RuntimeError(
            "AUTH_TOKEN no está definido. Configúralo como variable de entorno."
        )

    df_quiniela = descargar_historico(
        j_ini=JORNADA_INICIAL,
        j_fin=JORNADA_FINAL,
        temporada=NUM_TEMPORADA,
        uts=UTS_VALIDO
    )

    audit = auditoria_basica(df_quiniela, JORNADA_INICIAL, JORNADA_FINAL)

    print("\n===== AUDITORÍA =====")
    print(f"Filas: {audit['rows_total']} | Esperadas: {audit['expected_rows']} | OK: {audit['rows_ok']}")
    print(f"Jornadas faltantes: {audit['jornadas_faltantes']}")
    print(f"Jornadas incompletas: {audit['jornadas_incompletas']}")
    print(f"Duplicados (jornada,num): {audit['duplicados_por_jornada_num']}")
    print(f"Num fuera de 1..15: {audit['filas_num_fuera_rango']}")

    output_path = "data/quiniela_historico.csv"
    df_quiniela.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nCSV generado correctamente: {output_path}")

    return df_quiniela


# ============================================================
# CONFIG PROBABILIDADES
# ============================================================

BASE_URL = "https://api.eduardolosilla.es/probabilidades"
TEMPORADA = 2026
JORNADAS = range(1, 45)   # Ajusta a lo largo de la temporada
SLEEP_SECONDS_PROBABILIDADES = 0.4
OUTPUT_CSV = f"data/probabilidades_real_{TEMPORADA}.csv"

HEADERS_PROBABILIDADES = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.eduardolosilla.es/"
}


# ============================================================
# FUNCIONES PROBABILIDADES
# ============================================================

def fetch_jornada(session: requests.Session, temporada: int, jornada: int) -> dict:
    """
    Descarga el JSON de probabilidades para una jornada concreta.
    """
    params = {
        "num_temporada": temporada,
        "num_jornada": jornada
    }
    r = session.get(BASE_URL, params=params, headers=HEADERS_PROBABILIDADES, timeout=30)
    r.raise_for_status()
    return r.json()


def normalize_real(data: dict) -> pd.DataFrame:
    """
    Normaliza partidos.real.
    Maneja partidos 1–14 (1X2) y partido 15 (Pleno al 15).
    """
    temporada = int(data["numTemporada"])
    jornada = int(data["numJornada"])

    rows = []
    for item in data["partidos"]["real"]:
        base = {
            "numTemporada": temporada,
            "numJornada": jornada,
            "orden": item.get("orden"),
            "numero": item.get("numero")
        }

        # Partidos 1–14
        if "porc_1" in item:
            row = {
                **base,
                "tipo": "1X2",
                "porc_1": item.get("porc_1"),
                "porc_X": item.get("porc_X"),
                "porc_2": item.get("porc_2"),
                "porc_1_dec": item.get("porc_1_dec"),
                "porc_X_dec": item.get("porc_X_dec"),
                "porc_2_dec": item.get("porc_2_dec"),
                "delta_1": item.get("delta_1"),
                "delta_X": item.get("delta_X"),
                "delta_2": item.get("delta_2"),
            }
            rows.append(row)

        # Partido 15 (Pleno al 15)
        else:
            row = {
                **base,
                "tipo": "PLENO_15",
            }
            for k, v in item.items():
                if k.startswith("porc_15") or k.startswith("delta_15"):
                    row[k] = v
            rows.append(row)

    return pd.DataFrame(rows)


def run_probabilidades_scraper() -> pd.DataFrame:
    """
    Ejecuta el scraper de probabilidades y guarda el CSV.
    """
    all_dfs = []
    with requests.Session() as session:
        for j in JORNADAS:
            try:
                data = fetch_jornada(session, TEMPORADA, j)
                df_j = normalize_real(data)
                all_dfs.append(df_j)
                print(f"OK → Temporada {TEMPORADA}, Jornada {j}, filas: {len(df_j)}")
                time.sleep(SLEEP_SECONDS_PROBABILIDADES)
            except requests.HTTPError as e:
                print(f"HTTP ERROR en jornada {j}: {e}")
            except Exception as e:
                print(f"ERROR en jornada {j}: {e}")

    if not all_dfs:
        raise RuntimeError("No se ha podido extraer ninguna jornada.")

    df_prob = pd.concat(all_dfs, ignore_index=True)
    df_prob.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\nCSV generado correctamente: {OUTPUT_CSV}")

    return df_prob


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    run_escrutinios_scraper()
    run_probabilidades_scraper()
