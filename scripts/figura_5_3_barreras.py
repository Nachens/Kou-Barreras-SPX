"""Figura 5.3: grilla 2x2 (fecha x contrato): |V_Kou - V_BS| / s̄ contra la distancia x (la métrica de la hipótesis),
con una línea en 1. Con --metrica V, los dos precios en puntos del índice."""
import argparse

from _comun import RAIZ
from _fig_barreras import METRICAS, cargar, grilla

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", nargs="+", required=True)
    p.add_argument("--metrica", default="dif_sobre_sbar", choices=list(METRICAS))
    a = p.parse_args()
    datos, d_out = cargar(a.config)

    def dibujar(ax, d, ct):
        t = d["tab"][d["tab"].contrato == ct]
        S0 = d["cal"]["S0"]
        if a.metrica == "V":
            ax.plot(t.x, t.V_bs, "C3--", label="Black–Scholes")
            ax.plot(t.x, t.V_kou, "C0-", label="Kou")
        else:
            ax.plot(t.x, t[a.metrica], "C0.-", ms=3, label="|Kou − Black–Scholes| / $\\bar s$")
            if a.metrica == "dif_sobre_sbar":
                ax.axhline(1.0, color="0.4", ls=":", lw=1, label="$\\bar s$")
        ax.set_ylabel(METRICAS[a.metrica], fontsize=8)
        ax.legend(fontsize=7)

    plt, fig, axs = grilla(datos, dibujar)
    fig.tight_layout()
    f = d_out / "figura_5_3_barreras.pdf"
    fig.savefig(f)
    print(f)
