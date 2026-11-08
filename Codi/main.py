"""Execució (des de la carpeta P2/):  python codi/main.py
Opcions:  --pes_usuari 0.5   --sortida resultats/resultats.csv"""
import argparse
from pathlib import Path

import numpy as np

from carrega import carrega_matriu, carrega_parelles, escriu_resultats
from models import MitjanesUsuariRestaurant
from avaluacio import mae

ARREL = Path(__file__).resolve().parent.parent  # carpeta P2/


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dades", default=ARREL / "ConjuntDades_P2" / "recommendation_dataset.csv")
    ap.add_argument("--template", default=ARREL / "ConjuntDades_P2" / "template.csv")
    ap.add_argument("--target", default=ARREL / "ConjuntDades_P2" / "target_recommendations.csv")
    ap.add_argument("--sortida", default=ARREL / "Resultats" / "resultats.csv")
    ap.add_argument("--pes_usuari", type=float, default=0.5)
    args = ap.parse_args()

    D, idx_u, idx_r = carrega_matriu(args.dades)
    print(f"Matriu: {D.shape}, densitat = {np.mean(~np.isnan(D)):.3f}")

    usuaris, restaurants, _ = carrega_parelles(args.template, idx_u, idx_r)

    # Comprovació: les cel·les a predir han de ser desconegudes (99 -> NaN) a D
    frac_nan = np.mean(np.isnan(D[usuaris, restaurants]))
    print(f"Cel·les a predir que són NaN a D: {frac_nan:.3f} (hauria de ser 1.000)")

    model = MitjanesUsuariRestaurant(pes_usuari=args.pes_usuari).fit(D)
    prediccions = model.predict(usuaris, restaurants)

    target = Path(args.target)
    if target.exists():
        _, _, reals = carrega_parelles(target, idx_u, idx_r)
        print(f"MAE = {mae(reals, prediccions):.3f}")

    sortida = Path(args.sortida)
    sortida.parent.mkdir(parents=True, exist_ok=True)
    escriu_resultats(args.template, prediccions, sortida)
    print(f"Escrit {sortida}")


if __name__ == "__main__":
    main()