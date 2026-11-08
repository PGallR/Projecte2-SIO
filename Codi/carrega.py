"""Càrrega de dades: matriu D, parelles (usuari, restaurant) i escriptura de resultats."""
import numpy as np
import pandas as pd

VALOR_DESCONEGUT = 99.0
SEP = ";"


def carrega_matriu(path):
    """
    Llegeix recommendation_dataset.csv.
    Format: capçalera 'DATASET;Restaurant1;...' i files 'User1;v1;v2;...'.
    Retorna:
      D                -> matriu float32 (usuaris x restaurants), NaN = desconegut (99)
      idx_usuaris      -> pd.Index amb les etiquetes 'User1', 'User2', ...
      idx_restaurants  -> pd.Index amb les etiquetes 'Restaurant1', ...
    """
    cols = pd.read_csv(path, sep=SEP, nrows=0).columns[1:]
    df = pd.read_csv(
        path,
        sep=SEP,
        index_col=0,
        dtype={c: np.float32 for c in cols},
    )
    D = df.to_numpy(dtype=np.float32)
    D[D == VALOR_DESCONEGUT] = np.nan
    return D, df.index, df.columns


def carrega_parelles(path, idx_usuaris, idx_restaurants):
    """
    Llegeix template.csv o target_recommendations.csv (format 'User25287;Restaurant31;valor').
    Converteix les etiquetes a índexs de fila/columna de D buscant-les a idx_usuaris / idx_restaurants.
    Retorna (usuaris, restaurants, valors); 'valors' és None si la 3a columna és buida (template).
    """
    df = pd.read_csv(path, sep=SEP, header=None, names=["u", "r", "v"])
    usuaris = idx_usuaris.get_indexer(df["u"])
    restaurants = idx_restaurants.get_indexer(df["r"])
    if (usuaris < 0).any() or (restaurants < 0).any():
        raise ValueError(f"{path}: hi ha usuaris/restaurants que no existeixen a la matriu")
    valors = df["v"].to_numpy(dtype=np.float32) if df["v"].notna().all() else None
    return usuaris, restaurants, valors


def escriu_resultats(path_template, prediccions, path_sortida):
    """Escriu resultats.csv amb el mateix format que el template (separador ';', decimal '.')."""
    df = pd.read_csv(path_template, sep=SEP, header=None, names=["u", "r", "v"])
    df["v"] = np.round(prediccions.astype(np.float64), 4)
    df.to_csv(path_sortida, sep=SEP, header=False, index=False)