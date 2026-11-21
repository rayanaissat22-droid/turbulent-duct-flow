# -*- coding: utf-8 -*-
"""
main.py
=======
Point d'entrée : paramètres du projet et lancement des calculs.
Les résultats sont enregistrés dans resultats/*.npz ; les figures sont
produites séparément par traitement.py.

Exemples :
    python main.py laminaire
    python main.py turbulent --modele saturation
    python main.py balayage --modele parabolique
    python main.py convergence
    python main.py tout
"""

import argparse
import time

from canal_turbulent import MODELES, SimulationCanal, sauvegarder

# ============================================================
# PARAMÈTRES
# ============================================================

# Physique et modèle de turbulence (communs à tous les cas)
PHYSIQUE = dict(H=0.01, nu=1e-6, rho=1000.0, kappa=0.41, A_plus=26.0)

# Partie 1 : laminaire
LAMINAIRE = dict(dp_dx=-0.05, N=100, t_final=400.0)

# Partie 2 : cas de référence turbulent
TURBULENT = dict(dp_dx=-20.0, N=600, t_final=1000.0, tol=1e-4, dt_print=10.0)

# Balayage en Reynolds (une valeur de dp/dx par point du diagramme de Moody)
BALAYAGE = dict(
    dp_dx_liste=[-5.0, -10.0, -20.0, -50.0, -100.0, -150.0, -200.0, -250.0],
    t_final=500.0, tol=1e-5, z_plus_cible=1.0, N_min=300,
)

# Convergence : profils à différents instants
CONVERGENCE = dict(dp_dx=-20.0, N=600, tol=1e-9,
                   temps_sortie=[5.0, 10.0, 20.0, 30.0, 40.0])


# Utilisés quand main.py est lancé sans argument (par exemple depuis un IDE).
# Étape : "laminaire", "turbulent", "balayage", "convergence" ou "tout"
# Modèle : "saturation", "parabolique" ou "tous"
ETAPE_PAR_DEFAUT = "turbulent"
MODELE_PAR_DEFAUT = "parabolique"


# ============================================================
# ÉTAPES
# ============================================================

def entete(titre):
    print("\n" + "=" * 60)
    print(titre)
    print("=" * 60)


def etape_laminaire(modeles):
    entete("PARTIE 1 : ÉCOULEMENT LAMINAIRE")
    sim = SimulationCanal(dp_dx=LAMINAIRE["dp_dx"], N=LAMINAIRE["N"], **PHYSIQUE)
    donnees = sim.run_laminaire(t_final=LAMINAIRE["t_final"])
    sauvegarder("partie1.npz", donnees)


def etape_turbulent(modeles):
    for modele in modeles:
        entete(f"PARTIE 2 : ÉCOULEMENT TURBULENT — modèle {modele.upper()}")
        sim = SimulationCanal(dp_dx=TURBULENT["dp_dx"], N=TURBULENT["N"],
                              modele=modele, **PHYSIQUE)
        donnees = sim.run_turbulent(t_final=TURBULENT["t_final"],
                                    tol=TURBULENT["tol"],
                                    dt_print=TURBULENT["dt_print"])
        sauvegarder(f"partie2_{modele}.npz", donnees)


def etape_balayage(modeles):
    for modele in modeles:
        entete(f"BALAYAGE EN REYNOLDS — modèle {modele.upper()}")
        debut = time.perf_counter()
        donnees = SimulationCanal.balayage_reynolds(modele=modele,
                                                    **PHYSIQUE, **BALAYAGE)
        duree = time.perf_counter() - debut
        print(f"\n--- Temps réel total du balayage ({modele}) : "
              f"{duree:.1f} s ({duree/60:.2f} min) ---")
        sauvegarder(f"moody_{modele}.npz", donnees)


def etape_convergence(modeles):
    for modele in modeles:
        entete(f"CONVERGENCE DU PROFIL TURBULENT — modèle {modele.upper()}")
        sim = SimulationCanal(dp_dx=CONVERGENCE["dp_dx"], N=CONVERGENCE["N"],
                              modele=modele, **PHYSIQUE)
        donnees = sim.run_convergence(temps_sortie=CONVERGENCE["temps_sortie"],
                                      tol=CONVERGENCE["tol"])
        sauvegarder(f"convergence_{modele}.npz", donnees)


ETAPES = {
    "laminaire": etape_laminaire,
    "turbulent": etape_turbulent,
    "balayage": etape_balayage,
    "convergence": etape_convergence,
}


# ============================================================
# LANCEMENT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("etape", nargs="?", default=ETAPE_PAR_DEFAUT,
                        choices=[*ETAPES, "tout"],
                        help=f"calcul à lancer (défaut : {ETAPE_PAR_DEFAUT})")
    parser.add_argument("--modele", choices=[*MODELES, "tous"],
                        default=MODELE_PAR_DEFAUT,
                        help=f"modèle de longueur de mélange (défaut : {MODELE_PAR_DEFAUT})")
    args = parser.parse_args()

    modeles = MODELES if args.modele == "tous" else (args.modele,)
    etapes = ETAPES if args.etape == "tout" else {args.etape: ETAPES[args.etape]}

    for nom, fonction in etapes.items():
        debut = time.perf_counter()
        fonction(modeles)
        duree = time.perf_counter() - debut
        print(f"\nTemps réel d'exécution de l'étape '{nom}' : "
              f"{duree:.1f} s ({duree/60:.2f} min)")


if __name__ == "__main__":
    main()