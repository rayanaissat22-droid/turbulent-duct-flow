import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from canal_turbulent import DOSSIER_RESULTATS, MODELES

# Loi logarithmique de référence : u+ = ln(z+)/KAPPA_LOG + B_LOG
KAPPA_LOG = 0.41
B_LOG = 5.2

STYLES = {
    "saturation":  dict(label="Saturation",  marker="o", couleur="C0"),
    "parabolique": dict(label="Parabolique", marker="^", couleur="C1"),
}
STYLES_CONVERGENCE = [
    dict(marker="o", couleur="tab:blue"),
    dict(marker="s", couleur="tab:green"),
    dict(marker="^", couleur="tab:red"),
    dict(marker="v", couleur="tab:cyan"),
    dict(marker="D", couleur="tab:pink"),
]


# ============================================================
# Chargement
# ============================================================

def chemin(nom):
    return os.path.join(DOSSIER_RESULTATS, nom)


def charger(nom):
    """Charge un .npz s'il existe, renvoie None sinon."""
    return np.load(chemin(nom)) if os.path.exists(chemin(nom)) else None


def charger_modeles(prefixe):
    """{modele: données} pour les fichiers <prefixe>_<modele>.npz existants."""
    resultats = {}
    for modele in MODELES:
        donnees = charger(f"{prefixe}_{modele}.npz")
        if donnees is not None:
            resultats[modele] = donnees
    return resultats


def terminer(nom_figure, afficher):
    plt.tight_layout()
    plt.savefig(chemin(nom_figure), dpi=150)
    if afficher:
        plt.show()
    plt.close()


# ============================================================
# Résumé chiffré
# ============================================================

def resume(d1, ref, moody):
    print("=" * 60)
    print("RÉSUMÉ DES RÉSULTATS")
    print("=" * 60)

    if d1 is not None:
        print("\nPartie 1 (laminaire) :")
        print(f"  Re = {d1['Re']:.2f}")
        print(f"  Erreur relative max = {d1['erreur_relative']:.4f} %")

    for modele, d in ref.items():
        print(f"\nPartie 2, cas de référence, modèle {modele} :")
        print(f"  u_tau = {d['u_tau']:.6f} m/s")
        print(f"  U_m = {d['U_m']:.6f} m/s   U_c = {d['U_c']:.6f} m/s   "
              f"U_c/U_m = {float(d['U_c']) / float(d['U_m']):.4f}")
        print(f"  Re = {d['Re']:.2f}   f = {d['f']:.6f}")
        print(f"  Écart Blasius = {d['erreur_blasius']:.2f} %   "
              f"Écart Dean = {d['erreur_dean']:.2f} %")
        print(f"  Rapport de validation (bilan qdm) = {d['rapport_validation']:.6f}")

    for modele, m in moody.items():
        print(f"\nBalayage Reynolds, modèle {modele} : {len(m['Re_points'])} points, "
              f"Re de {m['Re_points'].min():.0f} à {m['Re_points'].max():.0f}")
        print(f"  Écart Dean : de {m['ecart_dean_points'][0]:+.2f} % "
              f"à {m['ecart_dean_points'][-1]:+.2f} %")


# ============================================================
# Figures
# ============================================================

def fig_laminaire(d1, afficher):
    plt.figure(figsize=(8, 5))
    plt.plot(d1["u_numerique"], d1["z"], "ob", markevery=4, label="Solution numérique")
    plt.plot(d1["u_analytique"], d1["z"], "-r", linewidth=2, label="Solution analytique")
    plt.xlabel("Vitesse u(z) [m/s]")
    plt.ylabel("Position verticale z [m]")
    plt.title("Profil de vitesse - écoulement laminaire")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    terminer("fig_partie1_profil.png", afficher)


def fig_profil_vitesse(ref, afficher):
    plt.figure(figsize=(7, 6))
    for modele, d in ref.items():
        plt.plot(d["u"], d["z"], linewidth=2, label=STYLES[modele]["label"])
    plt.xlabel("Vitesse moyenne u(z) [m/s]")
    plt.ylabel("Position z [m]")
    plt.title("Profil de vitesse turbulent")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    terminer("fig_partie2_profil_vitesse.png", afficher)


def fig_longueur_melange(ref, afficher):
    plt.figure(figsize=(7, 6))
    for modele, d in ref.items():
        plt.plot(d["lm"], d["z"], linewidth=2, label=STYLES[modele]["label"])
    plt.xlabel("Longueur de mélange lm [m]")
    plt.ylabel("Position z [m]")
    plt.title("Longueur de mélange de Van Driest")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    terminer("fig_partie2_longueur_melange.png", afficher)


def fig_variables_reduites(ref, afficher):
    plt.figure(figsize=(8, 6))
    for modele, d in ref.items():
        s = STYLES[modele]
        masque = d["z"] <= float(d["H"])          # moitié inférieure du canal
        plt.semilogx(d["z_plus"][masque], d["u_plus"][masque],
                     linestyle="none", marker=s["marker"], markersize=5,
                     markerfacecolor="none", color=s["couleur"], label=s["label"])

    z_visqueux = np.logspace(0, 1.5, 100)
    plt.semilogx(z_visqueux, z_visqueux, "--", label=r"$u^+=z^+$")
    z_log = np.logspace(1, 3.5, 200)
    plt.semilogx(z_log, np.log(z_log) / KAPPA_LOG + B_LOG, ":", linewidth=2,
                 label="Loi logarithmique")

    plt.xlabel(r"$z^+$")
    plt.ylabel(r"$u^+$")
    plt.title(r"Profil en variables réduites $u^+(z^+)$")
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.legend()
    terminer("fig_partie2_variables_reduites.png", afficher)


def fig_u_centre_temps(ref, afficher):
    plt.figure(figsize=(8, 5))
    for modele, d in ref.items():
        plt.plot(d["t_hist"], d["u_centre_hist"], linewidth=1.5,
                 label=STYLES[modele]["label"])
    plt.xlabel("Temps t [s]")
    plt.ylabel("Vitesse au centre u(H,t) [m/s]")
    plt.title("Évolution temporelle de la vitesse au centre du canal")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    terminer("fig_partie2_u_centre_temps.png", afficher)


def fig_moody(moody, afficher):
    Re_tous = np.concatenate([m["Re_points"] for m in moody.values()])
    Re_theorie = np.logspace(np.log10(Re_tous.min()) - 0.1,
                             np.log10(Re_tous.max()) + 0.1, 200)

    plt.figure(figsize=(8, 6))
    for modele, m in moody.items():
        s = STYLES[modele]
        plt.loglog(m["Re_points"], m["f_points"], s["marker"], markersize=8,
                   color=s["couleur"], label=f"Simulation ({modele})")
    plt.loglog(Re_theorie, 0.3164 / Re_theorie**0.25, "--",
               label="Blasius (conduite circulaire)")
    plt.loglog(Re_theorie, 0.292 / (Re_theorie / 2.0)**0.25, ":", linewidth=2,
               label="Dean (conduite plane)")
    plt.xlabel(r"$Re = U_m D_h / \nu$")
    plt.ylabel(r"Coefficient de frottement $f$")
    plt.title("Diagramme de Moody - conduite lisse")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    terminer("fig_moody.png", afficher)


def fig_ecarts(moody, afficher):
    plt.figure(figsize=(8, 6))
    for modele, m in moody.items():
        s = STYLES[modele]
        plt.semilogx(m["Re_points"], m["ecart_blasius_points"], linestyle="--",
                     marker=s["marker"], color=s["couleur"],
                     label=f"Écart Blasius ({modele})")
        plt.semilogx(m["Re_points"], m["ecart_dean_points"], linestyle="-",
                     marker=s["marker"], color=s["couleur"],
                     markerfacecolor="none", label=f"Écart Dean ({modele})")
    plt.axhline(0.0, color="gray", linewidth=1)
    plt.xlabel(r"$Re = U_m D_h / \nu$")
    plt.ylabel("Écart relatif avec la corrélation [%]")
    plt.title("Évolution de l'écart de f avec Blasius et Dean, en fonction de Re")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend(fontsize=9)
    terminer("fig_ecart_correlations.png", afficher)


def fig_variables_reduites_balayage(modele, m, afficher):
    """Tous les profils u+(z+) du balayage sur un même graphe."""
    n_cas = len(m["Re_points"])
    couleurs = plt.cm.viridis(np.linspace(0.0, 0.9, n_cas))
    marqueurs = ["o", "s", "^", "v", "D", "<", ">", "p", "h", "*"]

    plt.figure(figsize=(9, 6.5))
    for i in range(n_cas):
        z_plus = m[f"z_plus_{i}"]
        u_plus = m[f"u_plus_{i}"]
        # moitié inférieure du canal (z <= H) : maillage uniforme, donc la
        # première moitié des points
        n_moitie = (len(z_plus) - 1) // 2 + 1
        z_plus, u_plus = z_plus[:n_moitie], u_plus[:n_moitie]
        pas = max(1, len(z_plus) // 60)
        plt.semilogx(z_plus, u_plus, linestyle="-", linewidth=1.0,
                     color=couleurs[i], marker=marqueurs[i % len(marqueurs)],
                     markersize=5, markevery=pas, markerfacecolor="none",
                     label=f"Re = {m['Re_points'][i]:.0f}")

    z_visqueux = np.logspace(0, 1.5, 100)
    plt.semilogx(z_visqueux, z_visqueux, "k--", label=r"$u^+=z^+$")
    z_log = np.logspace(1, 3.5, 200)
    plt.semilogx(z_log, np.log(z_log) / KAPPA_LOG + B_LOG, "k:", linewidth=2,
                 label="Loi logarithmique")

    plt.xlabel(r"$z^+$")
    plt.ylabel(r"$u^+$")
    plt.title(rf"Balayage en Reynolds, profils $u^+(z^+)$ — modèle {modele}")
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.legend(fontsize=9)
    terminer(f"fig_balayage_variables_reduites_{modele}.png", afficher)


def fig_convergence(modele, d, afficher):
    plt.figure(figsize=(7, 7))
    for i, (t_val, profil) in enumerate(zip(d["temps_sortie"], d["profils"])):
        s = STYLES_CONVERGENCE[i % len(STYLES_CONVERGENCE)]
        plt.plot(profil, d["z"], linestyle="none", marker=s["marker"],
                 markersize=5, markerfacecolor="none", color=s["couleur"],
                 markevery=10, label=f"t_final = {int(t_val)} s")
    plt.xlabel("u (m/s)")
    plt.ylabel("z (m)")
    plt.title(f"Convergence du profil turbulent ({modele})")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    terminer(f"fig_convergence_profil_{modele}.png", afficher)


# ============================================================
# Exécution
# ============================================================

def run(afficher=True):
    os.makedirs(DOSSIER_RESULTATS, exist_ok=True)

    d1 = charger("partie1.npz")
    ref = charger_modeles("partie2")
    moody = charger_modeles("moody")
    conv = charger_modeles("convergence")

    if d1 is None and not ref and not moody and not conv:
        print(f"Aucun résultat trouvé dans '{DOSSIER_RESULTATS}/' : "
              "lancez d'abord main.py.")
        return

    resume(d1, ref, moody)

    if d1 is not None:
        fig_laminaire(d1, afficher)
    if ref:
        fig_profil_vitesse(ref, afficher)
        fig_longueur_melange(ref, afficher)
        fig_variables_reduites(ref, afficher)
        fig_u_centre_temps(ref, afficher)
    if moody:
        fig_moody(moody, afficher)
        fig_ecarts(moody, afficher)
        for modele, m in moody.items():
            if "z_plus_0" in m.files:
                fig_variables_reduites_balayage(modele, m, afficher)
            else:
                print(f"\nmoody_{modele}.npz ne contient pas les profils : "
                      f"relancez 'python main.py balayage --modele {modele}' "
                      "pour tracer u+(z+) du balayage.")
    for modele, d in conv.items():
        fig_convergence(modele, d, afficher)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sans-affichage", action="store_true",
                        help="enregistre les figures sans les afficher")
    args = parser.parse_args()
    run(afficher=not args.sans_affichage)