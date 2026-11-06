"""Figura 5.1: volatilidad implícita contra k = log(K/S0): bid-ask como barras, Black-Scholes y Kou (óptimos en C)."""
import numpy as np

from _comun import cargar_calibracion, cargar_cfg, fig_estilo, parser, preparar, salida
from bs.vol_implicita import vol_implicita

if __name__ == "__main__":
    a = parser("Figura 5.1").parse_args()
    cfg = cargar_cfg(a.config[0])
    cal = cargar_calibracion(cfg)
    cad, df, info, prob = preparar(cfg)
    plt = fig_estilo()
    k = np.log(prob.K / prob.S0)
    iv = lambda precios: np.array([vol_implicita(p_, t, prob.S0, K, prob.T, prob.r, prob.delta)
                                   for p_, t, K in zip(precios, prob.tipo, prob.K)])
    ivb, iva, ivm = iv(prob.bid), iv(prob.ask), prob.iv_mkt
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ok = np.isfinite(ivb) & np.isfinite(iva)
    ax.vlines(k[ok], ivb[ok], iva[ok], color="0.55", lw=3, label="bid–ask del mercado")
    ax.plot(k, ivm, "k.", ms=3, label="punto medio")
    ax.axhline(cal["bs"]["sigma_elegida"], color="C3", ls="--", label=f"Black–Scholes ($\\sigma$={cal['bs']['sigma_elegida']:.3f})")
    pts = cal["C"]["puntos"] if not cal["C"]["vacio"] else cal["kou"]["minimos"]
    etiqueta = "Kou (puntos de $\\mathcal{C}$)" if not cal["C"]["vacio"] else "Kou (mínimos locales; C vacío)"
    for i, p in enumerate(pts):
        ax.plot(k, iv(prob.precios_modelo(np.array(p["theta"]), cfg["calibracion"]["N_fft_final"])), color="C0", alpha=0.35,
                lw=1, label=etiqueta if i == 0 else None)
    ax.plot(k, iv(prob.precios_modelo(np.array(cal["kou"]["optimo"]["theta"]), cfg["calibracion"]["N_fft_final"])), color="C0", lw=2,
            label="Kou (óptimo global)")
    ax.set_xlabel("$k=\\log(K/S_0)$")
    ax.set_ylabel("volatilidad implícita")
    from _fig_barreras import titulo_fecha
    ax.set_title(titulo_fecha(cfg))
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(salida(cfg) / "figura_5_1_sonrisa.pdf")
    print(salida(cfg) / "figura_5_1_sonrisa.pdf")
