"""Figura 5.2: R_perf(lambda) con la línea del umbral de C."""
import numpy as np

from _comun import cargar_calibracion, cargar_cfg, fig_estilo, parser, salida

if __name__ == "__main__":
    a = parser("Figura 5.2").parse_args()
    cfg = cargar_cfg(a.config[0])
    cal = cargar_calibracion(cfg)
    plt = fig_estilo()
    lam = np.array([p["lam"] for p in cal["perfil"]])
    R = np.array([p["R"] for p in cal["perfil"]])
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(lam, R, "o-", ms=3)
    crit = cal["C"]["criterio"]
    if crit == "eps0":
        ax.axhline(cal["eps0_sq"], color="C3", ls="--", label="umbral de $\\mathcal{C}$: $\\varepsilon_0^2$")
    elif crit == "kappa":
        ax.axhline(cal["C"]["kappa"] ** 2 * cal["kou"]["optimo"]["R"], color="C3", ls="--", label="umbral de $\\mathcal{C}$: $\\kappa^2R_{\\min}$")
    else:
        ax.plot([], [], " ", label="criterio bid–ask: sin umbral en R")
    ax.axvline(cal["kou"]["optimo"]["theta"][1], color="0.5", ls=":", label="$\\lambda$ del óptimo")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$\\lambda$")
    ax.set_ylabel("$R_{\\mathrm{perf}}(\\lambda)$")
    from _fig_barreras import titulo_fecha
    ax.set_title(titulo_fecha(cfg))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(salida(cfg) / "figura_5_2_perfil.pdf")
    print(salida(cfg) / "figura_5_2_perfil.pdf")
