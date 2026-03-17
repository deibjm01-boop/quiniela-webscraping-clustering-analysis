import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================

def cargar_datos():
    """
    Carga los CSV generados por el scraping.
    """
    df_quiniela = pd.read_csv("data/quiniela_historico.csv")
    df_prob = pd.read_csv("data/probabilidades_real_2026.csv")
    return df_quiniela, df_prob


def preparar_probabilidades(df_prob: pd.DataFrame) -> pd.DataFrame:
    """
    Selecciona y renombra las columnas de probabilidades
    para facilitar el merge posterior.
    """
    cols_probs = [
        "numTemporada", "numJornada", "orden", "tipo",
        "porc_1", "porc_X", "porc_2",
        "porc_15L_0", "porc_15L_1", "porc_15L_2", "porc_15L_M",
        "porc_15V_0", "porc_15V_1", "porc_15V_2", "porc_15V_M"
    ]

    df_cols = df_prob[cols_probs].copy()

    df_cols = df_cols.rename(columns={
        "numTemporada": "num_temporada",
        "numJornada": "jornada_consultada",
        "orden": "num",

        "porc_1": "probabilidad1",
        "porc_X": "probabilidadX",
        "porc_2": "probabilidad2",

        "porc_15L_0": "probabilidad15L0",
        "porc_15L_1": "probabilidad15L1",
        "porc_15L_2": "probabilidad15L2",
        "porc_15L_M": "probabilidad15LM",

        "porc_15V_0": "probabilidad15V0",
        "porc_15V_1": "probabilidad15V1",
        "porc_15V_2": "probabilidad15V2",
        "porc_15V_M": "probabilidad15VM",
    })

    return df_cols


def unir_datasets(df_quiniela: pd.DataFrame, df_cols: pd.DataFrame) -> pd.DataFrame:
    """
    Une los datos históricos de quiniela con las probabilidades reales.
    """
    df = pd.merge(
        df_quiniela[
            [
                "num", "local", "visitante", "division", "dia", "hora",
                "porcentaje1", "porcentajeX", "porcentaje2",
                "resultado", "signo", "signo_goles", "id_besoccer",
                "temporada",
                "porcentaje15L0", "porcentaje15L1", "porcentaje15L2", "porcentaje15LM",
                "porcentaje15V0", "porcentaje15V1", "porcentaje15V2", "porcentaje15VM",
                "jornada_consultada", "num_temporada"
            ]
        ],
        df_cols,
        on=["num", "jornada_consultada", "num_temporada"],
        how="left",
        validate="one_to_one"
    )

    return df


def diff_prob_vs_pct_all(row):
    """
    Calcula las diferencias entre probabilidad real y porcentaje base
    para los tres signos (1, X, 2), sin depender del resultado.
    """
    if row.get("num") == 15:
        return np.nan, np.nan, np.nan

    return (
        row.get("probabilidad1") - row.get("porcentaje1")
        if pd.notna(row.get("probabilidad1")) and pd.notna(row.get("porcentaje1")) else np.nan,

        row.get("probabilidadX") - row.get("porcentajeX")
        if pd.notna(row.get("probabilidadX")) and pd.notna(row.get("porcentajeX")) else np.nan,

        row.get("probabilidad2") - row.get("porcentaje2")
        if pd.notna(row.get("probabilidad2")) and pd.notna(row.get("porcentaje2")) else np.nan
    )


def anadir_diferencias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade diff_1, diff_X y diff_2 al dataset.
    """
    df = df.copy()
    df[["diff_1", "diff_X", "diff_2"]] = df.apply(
        diff_prob_vs_pct_all,
        axis=1,
        result_type="expand"
    )
    return df


# ============================================================
# SELECCIÓN DE JORNADAS DE LIGA Y CREACIÓN DE VARIABLES
# ============================================================

def calcular_signo_probable(df: pd.DataFrame) -> pd.Series:
    """
    Calcula el signo con mayor probabilidad estimada.
    """
    return (
        df[["probabilidad1", "probabilidadX", "probabilidad2"]]
        .idxmax(axis=1)
        .map({
            "probabilidad1": "1",
            "probabilidadX": "X",
            "probabilidad2": "2"
        })
    )


def preparar_df_liga(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra las jornadas de liga y crea variables analíticas.
    """
    jornadas_liga = [1, 2, 3, 5, 7, 9, 11, 14, 16, 17, 19, 22, 24, 26, 28, 30, 32, 33, 35, 37, 39, 40, 42, 44]

    df_liga = df[df["jornada_consultada"].isin(jornadas_liga)].copy()

    equipos_primera = {
        "ALAVÉS",
        "ATH.CLUB",
        "AT.MADRID",
        "BARCELONA",
        "CELTA",
        "ELCHE",
        "ESPANYOL",
        "GETAFE",
        "GIRONA",
        "LEVANTE",
        "MALLORCA",
        "OSASUNA",
        "RAYO",
        "BETIS",
        "R.MADRID",
        "R.OVIEDO",
        "R.SOCIEDAD",
        "SEVILLA",
        "VALENCIA",
        "VILLARREAL"
    }

    # Aproximación de división basada en el equipo local
    df_liga["division"] = df_liga["local"].apply(
        lambda x: 1 if x in equipos_primera else 2
    )

    df_liga["signo_probable"] = calcular_signo_probable(df_liga)

    df_liga["acierto_signo_probable"] = (
        df_liga["signo_probable"] == df_liga["signo"]
    ).astype(int)

    return df_liga


def tipo_partido(row):
    """
    Clasifica el partido según la claridad del favorito.
    """
    probs = sorted(
        [row["probabilidad1"], row["probabilidadX"], row["probabilidad2"]],
        reverse=True
    )
    if probs[0] >= 60:
        return "favorito_claro"
    if probs[0] <= 40:
        return "equilibrado"
    return "intermedio"


def anadir_variables_analiticas(df_liga: pd.DataFrame) -> pd.DataFrame:
    """
    Añade variables analíticas auxiliares para el EDA.
    """
    df_liga = df_liga.copy()

    df_liga["tipo_partido"] = df_liga.apply(tipo_partido, axis=1)

    df_liga["max_diff"] = df_liga[["diff_1", "diff_X", "diff_2"]].abs().max(axis=1)

    df_liga["prob_signo_real"] = df_liga.apply(
        lambda r: r["probabilidad1"] if r["signo"] == "1"
        else r["probabilidadX"] if r["signo"] == "X"
        else r["probabilidad2"],
        axis=1
    )

    return df_liga


# ============================================================
# RESUMEN GENERAL DEL DATASET
# ============================================================

def resumen_general(df: pd.DataFrame):
    print("\n===== RESUMEN GENERAL =====")
    print(df.describe())
    print("\n===== CONTEO DE PARTIDOS POR NUM =====")
    print(df["num"].value_counts().sort_index())
    print("\n===== TIPO POR NUM =====")
    print(df.groupby("num")["tipo"].value_counts())


def validar_probabilidades(df: pd.DataFrame):
    print("\n===== VALIDACIÓN DE PROBABILIDADES 1X2 =====")
    mask_partidos_1x2 = df["num"].between(1, 14)
    print(
        df.loc[mask_partidos_1x2, ["probabilidad1", "probabilidadX", "probabilidad2"]]
        .sum(axis=1)
        .describe()
    )

    print("\n===== VALIDACIÓN PLENO AL 15 LOCAL =====")
    mask_partido_15 = df["num"] == 15
    print(
        df.loc[mask_partido_15, [
            "probabilidad15L0", "probabilidad15L1", "probabilidad15L2", "probabilidad15LM"
        ]].sum(axis=1).describe()
    )

    print("\n===== VALIDACIÓN PLENO AL 15 VISITANTE =====")
    print(
        df.loc[mask_partido_15, [
            "probabilidad15V0", "probabilidad15V1", "probabilidad15V2", "probabilidad15VM"
        ]].sum(axis=1).describe()
    )


# ============================================================
# ANÁLISIS DE FAVORITOS Y EMPATES
# ============================================================

def analisis_basico_signos(df_liga: pd.DataFrame):
    print("\n===== DISTRIBUCIÓN DE SIGNOS =====")
    print(df_liga["signo"].value_counts(normalize=True) * 100)

    mask_partidos_1x2 = df_liga["num"].between(1, 14)
    df_liga_1x2 = df_liga[mask_partidos_1x2].copy()

    print("\n===== ACIERTO DEL SIGNO PROBABLE =====")
    print((df_liga_1x2["signo_probable"] == df_liga_1x2["signo"]).mean())

    print("\n===== PARTIDOS CON MAYOR DIFERENCIA ENTRE PROBABILIDAD REAL Y PORCENTAJE BASE =====")
    print(
        df_liga.loc[df_liga["num"].between(1, 14)]
        .sort_values("max_diff", ascending=False)
        .head(10)[
            ["local", "visitante", "jornada_consultada",
             "diff_1", "diff_X", "diff_2", "signo"]
        ]
    )


def analisis_favoritos(df_liga: pd.DataFrame):
    df_70 = df_liga[
        df_liga["num"].between(1, 14) &
        (
            (df_liga["probabilidad1"] >= 70) |
            (df_liga["probabilidadX"] >= 70) |
            (df_liga["probabilidad2"] >= 70)
        )
    ].copy()

    df_70["gana_favorito"] = (
        ((df_70["probabilidad1"] >= 70) & (df_70["signo"] == "1")) |
        ((df_70["probabilidadX"] >= 70) & (df_70["signo"] == "X")) |
        ((df_70["probabilidad2"] >= 70) & (df_70["signo"] == "2"))
    )

    print("\n===== FAVORITOS >= 70 =====")
    print(df_70["gana_favorito"].value_counts())
    print(df_70["gana_favorito"].value_counts(normalize=True) * 100)

    df_50 = df_liga[
        df_liga["num"].between(1, 14) &
        (
            (df_liga["probabilidad1"] >= 50) |
            (df_liga["probabilidadX"] >= 50) |
            (df_liga["probabilidad2"] >= 50)
        )
    ].copy()

    df_50["gana_favorito"] = (
        ((df_50["probabilidad1"] >= 50) & (df_50["signo"] == "1")) |
        ((df_50["probabilidadX"] >= 50) & (df_50["signo"] == "X")) |
        ((df_50["probabilidad2"] >= 50) & (df_50["signo"] == "2"))
    )

    print("\n===== FAVORITOS >= 50 =====")
    print(df_50["gana_favorito"].value_counts())
    print(df_50["gana_favorito"].value_counts(normalize=True) * 100)


def analisis_empates(df_liga: pd.DataFrame):
    mask_x = (df_liga["probabilidadX"] > 30) & df_liga["num"].between(1, 14)

    print("\n===== SIGNOS REALES CUANDO probabilidadX > 30 =====")
    print(df_liga.loc[mask_x, "signo"].value_counts(normalize=True) * 100)


# ============================================================
# VISUALIZACIONES EDA
# ============================================================

def plot_porcentaje_base(df_liga: pd.DataFrame):
    df_plot_porcentajes = df_liga[df_liga["num"].between(1, 14)]

    bins = np.linspace(0, 100, 21)

    counts_1, _ = np.histogram(df_plot_porcentajes["porcentaje1"].dropna(), bins=bins)
    counts_X, _ = np.histogram(df_plot_porcentajes["porcentajeX"].dropna(), bins=bins)
    counts_2, _ = np.histogram(df_plot_porcentajes["porcentaje2"].dropna(), bins=bins)

    x = np.arange(len(counts_1))
    width = 0.25

    plt.figure(figsize=(12, 5))
    plt.bar(x - width, counts_1, width, label="1")
    plt.bar(x, counts_X, width, label="X")
    plt.bar(x + width, counts_2, width, label="2")

    plt.xlabel("Rango de porcentaje base")
    plt.ylabel("Número de partidos")
    plt.title("Conteo de porcentaje base en 20 binnings (1, X, 2)")
    plt.xticks(
        x,
        [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-1)],
        rotation=45
    )
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_diferencias_probabilidad(df: pd.DataFrame):
    df_plot_diffs = df[df["num"].between(1, 14)][["diff_1", "diff_X", "diff_2"]]

    plt.figure(figsize=(8, 5))
    plt.boxplot(
        [
            df_plot_diffs["diff_1"].dropna(),
            df_plot_diffs["diff_X"].dropna(),
            df_plot_diffs["diff_2"].dropna()
        ],
        labels=["diff_1", "diff_X", "diff_2"]
    )

    plt.axhline(0)
    plt.title("Diferencia entre probabilidad real y porcentaje base")
    plt.ylabel("Probabilidad real − Porcentaje base")
    plt.xlabel("Signo")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.hist(df_plot_diffs["diff_1"].dropna(), bins=30, alpha=0.7, label="diff_1")
    plt.hist(df_plot_diffs["diff_X"].dropna(), bins=30, alpha=0.7, label="diff_X")
    plt.hist(df_plot_diffs["diff_2"].dropna(), bins=30, alpha=0.7, label="diff_2")

    plt.axvline(0)
    plt.legend()
    plt.title("Distribución de diferencias probabilidad − porcentaje")
    plt.xlabel("Diferencia")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.show()


def stacked_signo_plot_auto_range(
    df,
    signo_objetivo: str,
    prob_col: str,
    titulo: str,
    bin_width: float = 2.5
):
    """
    Barra apilada (Sale / No sale) para un signo (1, X o 2),
    usando automáticamente el rango real de probabilidad del signo.
    """
    df_tmp = df[df["num"].between(1, 14)].copy()
    df_tmp["sale_signo"] = (df_tmp["signo"] == signo_objetivo).astype(int)
    df_tmp = df_tmp[df_tmp[prob_col].notna()]

    p_min = df_tmp[prob_col].min()
    p_max = df_tmp[prob_col].max()

    p_min = np.floor(p_min / bin_width) * bin_width
    p_max = np.ceil(p_max / bin_width) * bin_width

    bins = np.arange(p_min, p_max + bin_width, bin_width)

    df_tmp["bucket_prob"] = pd.cut(
        df_tmp[prob_col],
        bins=bins,
        right=False
    )

    counts = (
        df_tmp
        .groupby(["bucket_prob", "sale_signo"], observed=True)
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "No sale", 1: "Sale"})
    )

    ax = counts.plot(
        kind="bar",
        stacked=True,
        figsize=(13, 6),
        color=["#d3d3d3", "#2ca02c"],
        edgecolor="black"
    )

    for i, (_, row) in enumerate(counts.iterrows()):
        acumulado = 0
        for col in counts.columns:
            valor = row[col]
            if valor > 0:
                ax.text(
                    i,
                    acumulado + valor / 2,
                    f"{int(valor)}",
                    ha="center",
                    va="center",
                    fontsize=9
                )
            acumulado += valor

    ax.set_ylabel("Número de partidos")
    ax.set_xlabel("Probabilidad estimada (%)")
    ax.set_title(titulo)

    ax.set_xticklabels(
        [str(interval) for interval in counts.index],
        rotation=45,
        ha="right"
    )

    ax.legend(title="Resultado real")
    plt.tight_layout()
    plt.show()


def plot_fiabilidad_por_jornada(df_liga: pd.DataFrame):
    df_tmp = df_liga[df_liga["num"].between(1, 14)].copy()

    count_fav = (
        df_tmp
        .groupby(["jornada_consultada", "acierto_signo_probable"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={
            0: "No acierta favorito",
            1: "Acierta favorito"
        })
        .sort_index()
    )

    ax = count_fav.plot(
        kind="bar",
        stacked=True,
        figsize=(12, 6),
        color=["#d3d3d3", "#2ca02c"],
        edgecolor="black"
    )

    for i, (_, row) in enumerate(count_fav.iterrows()):
        acumulado = 0
        for col in count_fav.columns:
            valor = row[col]
            if valor > 0:
                ax.text(
                    i,
                    acumulado + valor / 2,
                    f"{int(valor)}",
                    ha="center",
                    va="center",
                    fontsize=9
                )
            acumulado += valor

    ax.set_xlabel("Jornada")
    ax.set_ylabel("Número de partidos")
    ax.set_title("Aciertos del favorito por jornada\n(signo con mayor probabilidad)")
    ax.legend(title="Resultado del favorito")
    plt.tight_layout()
    plt.show()


def plot_signos_por_jornada(df_liga: pd.DataFrame):
    df_tmp = df_liga[df_liga["num"].between(1, 14)].copy()

    count_signo = (
        df_tmp
        .groupby(["jornada_consultada", "signo"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    ax = count_signo.plot(
        kind="bar",
        stacked=True,
        figsize=(13, 6),
        color={
            "1": "#2ca02c",
            "X": "#1f77b4",
            "2": "#ff7f0e"
        },
        edgecolor="black"
    )

    for i, (_, row) in enumerate(count_signo.iterrows()):
        acumulado = 0
        for signo in count_signo.columns:
            valor = row[signo]
            if valor > 0:
                ax.text(
                    i,
                    acumulado + valor / 2,
                    f"{int(valor)}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if signo != "X" else "black"
                )
            acumulado += valor

    handles, labels = ax.get_legend_handles_labels()
    orden_deseado = ["X", "2", "1"]
    handles_ordenados = [handles[labels.index(o)] for o in orden_deseado]

    ax.legend(
        handles_ordenados,
        orden_deseado,
        title="Signo"
    )

    ax.set_xlabel("Jornada")
    ax.set_ylabel("Número de partidos")
    ax.set_title("Distribución de signos reales por jornada (1–X–2)")
    plt.tight_layout()
    plt.show()


def plot_empates_vs_fiabilidad(df_liga: pd.DataFrame):
    df_tmp = df_liga[df_liga["num"].between(1, 14)].copy()

    df_jornada_empates = (
        df_tmp.groupby("jornada_consultada")
        .apply(lambda x: pd.Series({
            "num_empates": (x["signo"] == "X").sum(),
            "acierto_favorito": x["acierto_signo_probable"].mean()
        }))
        .reset_index()
    )

    plt.figure(figsize=(7, 5))
    plt.scatter(
        df_jornada_empates["num_empates"],
        df_jornada_empates["acierto_favorito"],
        alpha=0.7
    )

    plt.xlabel("Número de empates en la jornada")
    plt.ylabel("Acierto del favorito")
    plt.title("Empates vs fiabilidad del favorito por jornada")
    plt.grid(True)
    plt.show()


def resumen_correlaciones_signos(df_liga: pd.DataFrame):
    df_tmp = df_liga[df_liga["num"].between(1, 14)].copy()

    df_jornada_signos = (
        df_tmp.groupby("jornada_consultada")
        .apply(lambda x: pd.Series({
            "num_X": (x["signo"] == "X").sum(),
            "num_1": (x["signo"] == "1").sum(),
            "num_2": (x["signo"] == "2").sum(),
            "acierto_favorito": x["acierto_signo_probable"].mean()
        }))
        .reset_index()
    )

    metricas = []

    for signo, col in {
        "X": "num_X",
        "1": "num_1",
        "2": "num_2"
    }.items():

        pearson = df_jornada_signos[col].corr(
            df_jornada_signos["acierto_favorito"],
            method="pearson"
        )

        spearman = df_jornada_signos[col].corr(
            df_jornada_signos["acierto_favorito"],
            method="spearman"
        )

        metricas.append({
            "signo": signo,
            "pearson_corr": pearson,
            "spearman_corr": spearman,
            "R2_aprox": pearson**2
        })

    df_metricas = pd.DataFrame(metricas).set_index("signo")
    print("\n===== CORRELACIONES SIGNOS VS FIABILIDAD DEL FAVORITO =====")
    print(df_metricas)


# ============================================================
# PIPELINE EDA
# ============================================================

def main():
    df_quiniela, df_prob = cargar_datos()
    df_cols = preparar_probabilidades(df_prob)
    df = unir_datasets(df_quiniela, df_cols)
    df = anadir_diferencias(df)

    df_liga = preparar_df_liga(df)
    df_liga = anadir_variables_analiticas(df_liga)

    resumen_general(df)
    validar_probabilidades(df)
    analisis_basico_signos(df_liga)
    analisis_favoritos(df_liga)
    analisis_empates(df_liga)

    plot_porcentaje_base(df_liga)
    plot_diferencias_probabilidad(df)
    stacked_signo_plot_auto_range(
        df=df_liga,
        signo_objetivo="X",
        prob_col="probabilidadX",
        titulo="Empates reales según probabilidad estimada (rango real)"
    )
    stacked_signo_plot_auto_range(
        df=df_liga,
        signo_objetivo="1",
        prob_col="probabilidad1",
        titulo="Victorias locales reales según probabilidad estimada (rango real)"
    )
    stacked_signo_plot_auto_range(
        df=df_liga,
        signo_objetivo="2",
        prob_col="probabilidad2",
        titulo="Victorias visitantes reales según probabilidad estimada (rango real)"
    )
    plot_fiabilidad_por_jornada(df_liga)
    plot_signos_por_jornada(df_liga)
    plot_empates_vs_fiabilidad(df_liga)
    resumen_correlaciones_signos(df_liga)

    return df, df_liga


if __name__ == "__main__":
    main()
