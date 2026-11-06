"""Figura 5.4: por fecha y contrato, probabilidad de cruce por continuidad y por salto contra x (Kou calibrado)."""
import argparse

from _fig_barreras import cargar, grilla

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", nargs="+", required=True)
    a = p.parse_args()
    datos, d_out = cargar(a.config)

    def dibujar(ax, d, ct):
        t = d["tab"][d["tab"].contrato == ct]
        ax.stackplot(t.x, t.q_cont, t.q_salto, labels=["por continuidad ($Z=0$)", "por salto ($Z>0$)"],
                     colors=["C0", "C1"], alpha=0.8)
        ax.set_ylabel("probabilidad bajo Q de cruzar antes de T", fontsize=8)
        ax2 = ax.twinx()
        ax2.plot(t.x, t.frac_salto, "k:", lw=1.2, label="fracción por salto")
        ax2.set_ylim(0, 1)
        ax2.grid(False)
        ax.legend(fontsize=7, loc="upper left")
        ax2.legend(fontsize=7, loc="upper right")

    plt, fig, axs = grilla(datos, dibujar)
    fig.tight_layout()
    f = d_out / "figura_5_4_cruce.pdf"
    fig.savefig(f)
    print(f)
