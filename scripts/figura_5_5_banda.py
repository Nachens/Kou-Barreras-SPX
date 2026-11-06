"""Figura 5.5: como la 5.3 pero con la banda de Kou sobre C (cota inferior del rango sobre C)."""
import argparse

from _fig_barreras import METRICAS, cargar, grilla

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", nargs="+", required=True)
    a = p.parse_args()
    datos, d_out = cargar(a.config)

    def dibujar(ax, d, ct):
        t = d["tab"][d["tab"].contrato == ct]
        S0 = d["cal"]["S0"]
        ax.plot(t.x, t.V_bs, "C3--", label="Black–Scholes")
        ax.plot(t.x, t.V_kou, "C0-", label="Kou (óptimo global)")
        if d["banda"] is not None:
            b = d["banda"][d["banda"].contrato == ct]
            ax.fill_between(b.x, b.V_min, b.V_max, color="C0", alpha=0.25, label="banda de Kou sobre $\\mathcal{C}$")
        else:
            ax.text(0.05, 0.9, "C vacío: sin banda", transform=ax.transAxes, fontsize=8)
        ax.set_ylabel(METRICAS["V"], fontsize=8)
        ax.legend(fontsize=7)

    plt, fig, axs = grilla(datos, dibujar)
    fig.tight_layout()
    f = d_out / "figura_5_5_banda.pdf"
    fig.savefig(f)
    print(f)
