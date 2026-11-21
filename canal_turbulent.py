import os
import time
from dataclasses import dataclass, field

import numpy as np

DOSSIER_RESULTATS = "resultats"
MODELES = ("saturation", "parabolique")

# np.trapezoid n'existe que pour numpy >= 2.0 (np.trapz avant)
_integrale = getattr(np, "trapezoid", None) or np.trapz


def sauvegarder(nom_fichier, donnees):
    """Sauvegarde un dictionnaire dans resultats/<nom_fichier> (format .npz)."""
    os.makedirs(DOSSIER_RESULTATS, exist_ok=True)
    chemin = os.path.join(DOSSIER_RESULTATS, nom_fichier)
    np.savez(chemin, **donnees)
    return chemin


@dataclass
class ResultatRANS:
    """Sortie brute du solveur RANS."""
    u: np.ndarray
    lm: np.ndarray
    nu_t: np.ndarray
    u_tau: float
    t: float
    n_pas: int
    t_hist: np.ndarray
    u_centre_hist: np.ndarray
    converge: bool
    profils: dict = field(default_factory=dict)   # instantanés : {temps: u(z)}


class SimulationCanal:
    """Canal plan de demi-hauteur H, maillage uniforme de N intervalles."""

    def __init__(self, H=0.01, nu=1e-6, rho=1000.0, dp_dx=-20.0, N=600,
                 kappa=0.41, A_plus=26.0, modele="saturation",
                 coef_saturation=0.09):
        if modele not in MODELES:
            raise ValueError(f"modèle inconnu : {modele!r} (choix : {MODELES})")

        self.H = H
        self.nu = nu
        self.rho = rho
        self.dp_dx = dp_dx
        self.K = -dp_dx / rho
        self.kappa = kappa
        self.A_plus = A_plus
        self.modele = modele
        self.lm_max = coef_saturation * H       # utilisé par "saturation"

        self.N = int(N)
        self.z = np.linspace(0.0, 2.0 * H, self.N + 1)
        self.dz = self.z[1] - self.z[0]
        self.d = np.minimum(self.z, 2.0 * H - self.z)   # distance à la paroi

    # ------------------------------------------------------------------
    # Quantités analytiques
    # ------------------------------------------------------------------
    @property
    def reynolds_laminaire(self):
        """Re = U_m D_h / nu pour l'écoulement de Poiseuille (D_h = 4H)."""
        return 4.0 * self.H**3 * (-self.dp_dx) / (3.0 * self.rho * self.nu**2)

    def poiseuille(self):
        """Solution analytique stationnaire : u = K/(2 nu) z (2H - z)."""
        return (self.K / (2.0 * self.nu)) * self.z * (2.0 * self.H - self.z)

    # ------------------------------------------------------------------
    # Partie 1 : laminaire
    # ------------------------------------------------------------------
    def resoudre_laminaire(self, t_final=400.0, coef_cfl=0.8, verbose=True):
        """du/dt = K + nu d²u/dz², Euler explicite, différences centrées."""
        dz, nu, K = self.dz, self.nu, self.K

        dt_max = 0.5 * dz**2 / nu
        dt = coef_cfl * dt_max
        n_pas = int(np.ceil(t_final / dt))

        if verbose:
            print(f"dz = {dz:.4e} m")
            print(f"dt_max = {dt_max:.4e} s")
            print(f"dt utilisé = {dt:.4e} s")
            print(f"Nombre d'itérations = {n_pas}")

        u = np.zeros_like(self.z)
        for _ in range(n_pas):
            u[1:-1] += dt * (K + nu * (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dz**2)
        return u

    def run_laminaire(self, t_final=400.0, coef_cfl=0.8, verbose=True):
        """Résout, compare à Poiseuille et renvoie un dict prêt à sauvegarder."""
        Re = self.reynolds_laminaire
        if verbose:
            print(f"Re = {Re:.2f}")
            print("Régime laminaire : OK" if Re < 1000 else "Attention : Re >= 1000")

        u_num = self.resoudre_laminaire(t_final, coef_cfl, verbose)
        u_ana = self.poiseuille()

        erreur_max = np.max(np.abs(u_num - u_ana))
        u_max = np.max(np.abs(u_ana))
        erreur_relative = erreur_max / u_max * 100.0 if u_max > 0 else 0.0

        if verbose:
            print(f"Erreur absolue maximale = {erreur_max:.4e} m/s")
            print(f"Erreur relative maximale = {erreur_relative:.4f} %")
            print(f"Vitesse maximale analytique = {np.max(u_ana):.6f} m/s")
            print(f"Vitesse maximale numérique  = {np.max(u_num):.6f} m/s")

        return dict(z=self.z, u_numerique=u_num, u_analytique=u_ana, Re=Re,
                    erreur_relative=erreur_relative, erreur_max=erreur_max)

    # ------------------------------------------------------------------
    # Partie 2 : turbulent, longueur de mélange
    # ------------------------------------------------------------------
    def longueur_melange(self, u_tau):
        """Longueur de mélange lm(z) selon le modèle choisi."""
        if u_tau > 0.0:
            d_plus = self.d * u_tau / self.nu
        else:
            d_plus = np.zeros_like(self.d)
        amortissement = 1.0 - np.exp(-d_plus / self.A_plus)      # Van Driest

        if self.modele == "saturation":
            return np.minimum(self.kappa * self.d * amortissement, self.lm_max)
        # modèle "parabolique" : pas de saturation
        return (self.kappa * self.d * (1.0 - self.d / (2.0 * self.H))
                * amortissement)

    def resoudre_rans(self, t_final=1000.0, tol=1e-4, dt_print=10.0,
                      temps_sortie=None, coef_dt=0.4, n_save=200,
                      verbose=True):
        """
        Résolution instationnaire de l'équation RANS, départ au repos.

        Arrêt quand max|Δu|/dt / K < tol, ou à t = t_final.
        Si temps_sortie est fourni, un instantané de u(z) est enregistré
        à chacun de ces temps et t_final = max(temps_sortie).
        dt_print = None supprime l'affichage de progression.
        """
        z, dz, K, nu = self.z, self.dz, self.K, self.nu
        Nz = len(z)

        if temps_sortie is not None:
            temps_sortie = sorted(temps_sortie)
            t_final = temps_sortie[-1]

        u = np.zeros(Nz)
        lm = np.zeros(Nz)
        nu_t = np.zeros(Nz)
        u_tau = 0.0
        t = 0.0
        n_pas = 0
        taux = np.inf
        converge = False
        t_print = dt_print

        i_centre = Nz // 2
        t_hist = [0.0]
        u_centre_hist = [u[i_centre]]

        profils = {}
        i_suivant = 0

        while t < t_final:
            u_old = u.copy()

            # vitesse de frottement d'après le gradient à la paroi
            du_dz_paroi = (u_old[1] - u_old[0]) / dz
            u_tau = np.sqrt(max(nu * abs(du_dz_paroi), 0.0))

            lm = self.longueur_melange(u_tau)

            du_dz = np.zeros(Nz)
            du_dz[1:-1] = (u_old[2:] - u_old[:-2]) / (2.0 * dz)
            nu_t = lm**2 * np.abs(du_dz)
            nu_eff = nu + nu_t

            # pas de temps adapté à la viscosité effective maximale
            dt = coef_dt * dz**2 / (2.0 * np.max(nu_eff))
            dt = min(dt, t_final - t)

            # diffusion en flux aux interfaces
            nu_plus = 0.5 * (nu_eff[1:-1] + nu_eff[2:])
            nu_moins = 0.5 * (nu_eff[1:-1] + nu_eff[:-2])
            flux_plus = nu_plus * (u_old[2:] - u_old[1:-1]) / dz
            flux_moins = nu_moins * (u_old[1:-1] - u_old[:-2]) / dz
            diffusion = (flux_plus - flux_moins) / dz

            u[1:-1] = u_old[1:-1] + dt * (K + diffusion)
            u[0] = 0.0
            u[-1] = 0.0

            t += dt
            n_pas += 1

            if n_pas % n_save == 0:
                t_hist.append(t)
                u_centre_hist.append(u[i_centre])

            # instantanés demandés
            if temps_sortie is not None:
                while i_suivant < len(temps_sortie) and t >= temps_sortie[i_suivant]:
                    profils[temps_sortie[i_suivant]] = u.copy()
                    if verbose:
                        print(f"Instantané capturé à t = {t:.3f} s "
                              f"(demandé : {temps_sortie[i_suivant]} s), {n_pas} pas")
                    i_suivant += 1

            # critère de convergence
            taux = np.max(np.abs(u - u_old)) / dt
            if taux / K < tol:
                converge = True
                t_hist.append(t)
                u_centre_hist.append(u[i_centre])
                if verbose:
                    print(f"Convergence atteinte à t = {t:.2f} s ({n_pas} pas de temps)")
                # instantanés restants : le profil est déjà stationnaire
                if temps_sortie is not None:
                    while i_suivant < len(temps_sortie):
                        profils[temps_sortie[i_suivant]] = u.copy()
                        i_suivant += 1
                break

            if verbose and dt_print is not None and t >= t_print:
                print(f"t = {t:7.2f} s   taux/K = {taux / K:.2e}")
                t_print += dt_print
        else:
            if verbose:
                print(f"t_final = {t_final} s atteint sans convergence "
                      f"(taux/K = {taux / K:.2e}).")

        return ResultatRANS(u=u, lm=lm, nu_t=nu_t, u_tau=u_tau, t=t, n_pas=n_pas,
                            t_hist=np.array(t_hist),
                            u_centre_hist=np.array(u_centre_hist),
                            converge=converge, profils=profils)

    def grandeurs_integrales(self, u, u_tau):
        """U_m, Re, coefficient de frottement f et écarts aux corrélations."""
        H, nu, rho = self.H, self.nu, self.rho

        U_m = _integrale(u, self.z) / (2.0 * H)
        Re = U_m * 4.0 * H / nu                       # basé sur D_h = 4H
        tau_w = rho * u_tau**2
        f = 8.0 * tau_w / (rho * U_m**2) if U_m > 0.0 else np.nan

        f_blasius = 0.3164 / Re**0.25 if Re > 0.0 else np.nan
        Re_2H = U_m * 2.0 * H / nu                    # basé sur 2H
        f_dean = 0.292 / Re_2H**0.25 if Re_2H > 0.0 else np.nan

        erreur_blasius = abs(f - f_blasius) / f_blasius * 100.0
        erreur_dean = abs(f - f_dean) / f_dean * 100.0

        # bilan de quantité de mouvement : u_tau² doit valoir K H
        # (facteur (N-1)/N conservé du code d'origine)
        rapport_validation = u_tau**2 / (self.K * H * (self.N - 1) / self.N)

        return dict(
            U_m=U_m, U_c=u[self.N // 2], Re=Re, f=f,
            f_blasius=f_blasius, f_dean=f_dean,
            erreur_blasius=erreur_blasius, erreur_dean=erreur_dean,
            rapport_validation=rapport_validation,
            z_plus=self.z * u_tau / nu,
            u_plus=(u / u_tau) if u_tau > 0.0 else np.zeros_like(u),
        )

    def run_turbulent(self, t_final=1000.0, tol=1e-4, dt_print=10.0, verbose=True):
        """Cas de référence : résout, calcule f et renvoie un dict à sauvegarder."""
        res = self.resoudre_rans(t_final=t_final, tol=tol, dt_print=dt_print,
                                 verbose=verbose)
        g = self.grandeurs_integrales(res.u, res.u_tau)

        if verbose:
            print(f"u_tau = {res.u_tau:.6f} m/s   U_m = {g['U_m']:.6f} m/s   "
                  f"U_c = {g['U_c']:.6f} m/s")
            print(f"U_c/U_m = {g['U_c'] / g['U_m']:.4f}")
            print(f"Re = {g['Re']:.2f}   f = {g['f']:.6f}")
            print(f"Écart Blasius = {g['erreur_blasius']:.2f} %   "
                  f"Écart Dean = {g['erreur_dean']:.2f} %")
            print(f"Rapport de validation = {g['rapport_validation']:.6f}")

        return dict(z=self.z, u=res.u, lm=res.lm,
                    t_hist=res.t_hist, u_centre_hist=res.u_centre_hist,
                    u_tau=res.u_tau, H=self.H, modele=self.modele, **g)

    # ------------------------------------------------------------------
    # Convergence : instantanés du profil
    # ------------------------------------------------------------------
    def run_convergence(self, temps_sortie=(5.0, 10.0, 20.0, 30.0, 40.0),
                        tol=1e-9, verbose=True):
        """Profils u(z) à des temps donnés (tol très strict : pas d'arrêt anticipé)."""
        temps = sorted(temps_sortie)
        res = self.resoudre_rans(tol=tol, dt_print=None, temps_sortie=temps,
                                 verbose=verbose)
        if verbose:
            print(f"\nCalcul terminé en {res.n_pas} pas de temps au total.")
        return dict(z=self.z, temps_sortie=np.array(temps),
                    profils=np.array([res.profils[t] for t in temps]),
                    H=self.H, modele=self.modele, n_pas=res.n_pas)

    # ------------------------------------------------------------------
    # Balayage en Reynolds (diagramme de Moody)
    # ------------------------------------------------------------------
    @classmethod
    def balayage_reynolds(cls, dp_dx_liste, modele="saturation",
                          H=0.01, nu=1e-6, rho=1000.0, kappa=0.41, A_plus=26.0,
                          t_final=500.0, tol=1e-5, z_plus_cible=1.0, N_min=300,
                          verbose=True):
        """
        Un calcul par valeur de dp/dx, maillage adapté pour que le premier
        point soit à z+ <= z_plus_cible (avec au moins N_min intervalles).
        """
        Re_l, f_l, fb_l, fd_l, eb_l, ed_l, duree_l, N_l = ([] for _ in range(8))
        profils_plus = {}          # z_plus_<i>, u_plus_<i> : un profil par cas

        for i, dp_dx in enumerate(dp_dx_liste):
            K = -dp_dx / rho
            u_tau0 = np.sqrt(K * H)                   # estimation de u_tau
            N = max(int(np.ceil(2.0 * H * u_tau0 / (nu * z_plus_cible))), N_min)

            sim = cls(H=H, nu=nu, rho=rho, dp_dx=dp_dx, N=N,
                      kappa=kappa, A_plus=A_plus, modele=modele)

            if verbose:
                print(f"\n--- dp/dx = {dp_dx:.1f} Pa/m  |  N = {N}  |  "
                      f"u_tau0 estimé = {u_tau0:.5f} m/s ---")

            debut = time.perf_counter()
            res = sim.resoudre_rans(t_final=t_final, tol=tol, dt_print=None,
                                    verbose=verbose)
            duree = time.perf_counter() - debut
            g = sim.grandeurs_integrales(res.u, res.u_tau)

            if verbose:
                print(f"    -> Re = {g['Re']:.1f}, f = {g['f']:.5f}, "
                      f"rapport validation = {g['rapport_validation']:.6f}, "
                      f"{res.n_pas} pas, temps réel = {duree:.1f} s ({duree/60:.2f} min)")
                print(f"       f_Blasius = {g['f_blasius']:.5f} "
                      f"(écart {g['erreur_blasius']:+.2f} %)   "
                      f"f_Dean = {g['f_dean']:.5f} (écart {g['erreur_dean']:+.2f} %)")

            Re_l.append(g["Re"]); f_l.append(g["f"])
            fb_l.append(g["f_blasius"]); fd_l.append(g["f_dean"])
            eb_l.append(g["erreur_blasius"]); ed_l.append(g["erreur_dean"])
            duree_l.append(duree); N_l.append(N)
            profils_plus[f"z_plus_{i}"] = g["z_plus"]
            profils_plus[f"u_plus_{i}"] = g["u_plus"]

        return dict(dp_dx_test=np.array(dp_dx_liste),
                    Re_points=np.array(Re_l), f_points=np.array(f_l),
                    f_blasius_points=np.array(fb_l), f_dean_points=np.array(fd_l),
                    ecart_blasius_points=np.array(eb_l),
                    ecart_dean_points=np.array(ed_l),
                    duree_points=np.array(duree_l), N_points=np.array(N_l),
                    modele=modele, H=H, **profils_plus)