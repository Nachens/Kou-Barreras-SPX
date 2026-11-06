"""Controles numéricos de la Tabla 5.4 . Cada función devuelve una lista de dicts
(control, que, error_max, tolerancia, ok).
"""
from __future__ import annotations

import numpy as np

from bs import barreras_rr as rr
from . import mc
from .barreras import Barreras, Resolvente
from .europeas import CarrMadan, call_gil_pelaez, prob_gil_pelaez
from .modelo import Kou, kou_Q

_NOTA9 = dict(eta1=10.0, eta2=5.0, lam=1.0, p=0.4, sigma=0.16, r=0.05, S0=100.0, K=98.0, T=0.5)
_KW = dict(sigma=0.2, p=0.5, eta1=50.0, eta2=1 / 0.03)
# Valores calculados con un prototipo previo (en `decimal`, 45 dígitos, n = 14). Solo la primera fila tiene valores
# publicados (KW_PUBLICADO); las otras tres son del prototipo.
KW_FILAS = [  # gamma, lambda, P(tau<=1), P(tau<=1, X_1>=0.2)
    (0.1, 3.0, 0.2558430, 0.2236155), (0.1, 0.01, 0.2605981, 0.2327533),
    (-0.1, 3.0, 0.0612228, 0.0439674), (-0.1, 0.01, 0.0581617, 0.0432472)]
KW_PUBLICADO = (0.25584, 0.22362)


def _fila(control, que, err, tol):
    return dict(control=control, que=que, error_max=float(err), tolerancia=float(tol), ok=bool(err <= tol))


# --------------------------------------------------------------------- lambda -> 0
def casos_lambda0(n_casos=6, semilla=1):
    """Grilla (S0 = 100, K, L, T, sigma, r, delta) fija por semilla; K y L compatibles con cada contrato."""
    rng = np.random.default_rng(semilla)
    casos = []
    for i in range(n_casos):
        casos.append(dict(S0=100.0, K=float(rng.uniform(92, 108)), T=float(rng.uniform(0.15, 1.0)),
                          sigma=float(rng.uniform(0.12, 0.35)), r=float(rng.uniform(0.0, 0.05)),
                          delta=float(rng.uniform(0.0, 0.03)), Lu=float(rng.uniform(112, 140)),
                          Ld=float(rng.uniform(62, 88))))
    return casos


def control_lambda0(casos=None, lams=(1e-2, 1e-4, 1e-6), n=14, B=2, dps=45):
    casos = casos or casos_lambda0()
    errs = {l: [] for l in lams}
    mono = True
    for c in casos:
        rr_u = rr.rr_up_out_call(c["S0"], c["K"], c["Lu"], c["T"], c["r"], c["delta"], c["sigma"])
        rr_d = rr.rr_down_out_put(c["S0"], c["K"], c["Ld"], c["T"], c["r"], c["delta"], c["sigma"])
        prev = (np.inf, np.inf)
        for lam in lams:
            kq = kou_Q(c["r"], c["delta"], c["sigma"], lam, 0.3, 25.0, 10.0)
            b = Barreras(kq, c["S0"], c["K"], c["T"], c["r"], c["delta"], n, B, dps)
            eu = float(b.call_up(c["Lu"])["v_out_inv"]) - rr_u
            ed = float(b.put_down(c["Ld"])["v_out_inv"]) - rr_d
            errs[lam].append(max(abs(eu), abs(ed)))
            cur = (abs(eu), abs(ed))
            mono &= cur[0] <= prev[0] * 1.0001 + 1e-9 and cur[1] <= prev[1] * 1.0001 + 1e-9
            prev = cur
    ultimo = max(errs[lams[-1]])
    return [_fila("λ→0 contra Rubinstein–Reiner", f"call up-and-out y put down-and-out, {len(casos)} casos; λ={lams[-1]:g}",
                  ultimo, 5e-6),
            _fila("λ→0: convergencia monótona", f"errores decrecen en λ ∈ {lams}", 0.0 if mono else 1.0, 0.0)], errs


# ------------------------------------------------------------------------ a = b
def control_a_igual_b(n=14, B=2, dps=45, semilla=2, n_casos=6):
    rng = np.random.default_rng(semilla)
    err = 0.0
    for _ in range(n_casos):
        k = Kou(rng.uniform(-0.1, 0.1), rng.uniform(0.1, 0.3), rng.uniform(0.3, 3), rng.uniform(0.2, 0.8),
                rng.uniform(5, 40), rng.uniform(4, 25))
        T = float(rng.uniform(0.2, 1.0))
        b = float(rng.uniform(0.05, 0.4))
        R = Resolvente(k, T, n, B, dps)
        err = max(err, abs(float(R.prob_joint(b, b)) - prob_gil_pelaez(k, T, b)))
    # y -> 0 cuando b -> infinito
    k = Kou(0.05, 0.2, 1.0, 0.4, 10.0, 5.0)
    R = Resolvente(k, 0.5, n, B, dps)
    cola = float(R.prob_joint(6.0, 6.0))
    return [_fila("a = b contra la marginal", f"forma invertida con a=b contra P(X_T≥b) por Gil-Pelaez, {n_casos} casos",
                  err, 1e-6),
            _fila("a = b: cola", "P(X_T≥b, τ_b≤T) en b=6 (debe → 0)", abs(cola), 1e-6)]


# ------------------------------------------------------------------- Kou y Wang
def control_kou_wang(n=14, B=2, dps=45):
    out = []
    e_nuestro = 0.0
    for gam, lam, pt, pj in KW_FILAS:
        R = Resolvente(Kou(gam, _KW["sigma"], lam, _KW["p"], _KW["eta1"], _KW["eta2"]), 1.0, n, B, dps)
        pc, pjp = R.prob_cruce(0.3)
        e_nuestro = max(e_nuestro, abs(float(pc + pjp) - pt), abs(float(R.prob_joint(0.2, 0.3)) - pj))
    R = Resolvente(Kou(0.1, _KW["sigma"], 3.0, _KW["p"], _KW["eta1"], _KW["eta2"]), 1.0, n, B, dps)
    pc, pjp = R.prob_cruce(0.3)
    e_pub = max(abs(float(pc + pjp) - KW_PUBLICADO[0]), abs(float(R.prob_joint(0.2, 0.3)) - KW_PUBLICADO[1]))
    out.append(_fila("Kou y Wang, Tablas 1–2 (publicado)", "γ=0,1, λ=3 contra 0,25584 y 0,22362", e_pub, 5e-6))
    out.append(_fila("Kou y Wang, cuatro filas (nuestros)", "contra los valores de un prototipo previo (solo la primera fila es publicada)", e_nuestro, 1e-6))
    return out


# ------------------------------------------------------------------- europeas
def control_europeas():
    n9 = _NOTA9
    kq = kou_Q(n9["r"], 0.0, n9["sigma"], n9["lam"], n9["p"], n9["eta1"], n9["eta2"])
    c9 = float(call_gil_pelaez(kq, n9["S0"], n9["K"], n9["T"], n9["r"], 0.0))
    e_nota = abs(c9 - 9.14732)
    kq2 = kou_Q(0.02, 0.015, 0.15, 1.5, 0.3, 25.0, 10.0)
    cm = CarrMadan(kq2, 0.25, 0.02, 0.015)
    K = np.arange(70.0, 140.0, 2.5)
    fft_ = cm.call(100.0, K)
    cuad = cm.call_cuadratura(100.0, K)
    e_fq = float(np.max(np.abs(fft_ - cuad) / np.maximum(cuad, 1e-8)))
    gp = np.array([call_gil_pelaez(kq2, 100.0, k_, 0.25, 0.02, 0.015) for k_ in K])
    e_gp = float(np.max(np.abs(fft_ - gp)))
    b = Barreras(kq2, 100.0, 100.0, 0.25, 0.02, 0.015, 14, 2, 45)
    e_inv = abs(float(b.call_up(115.0)["europea_inv"]) - float(cm.call(100.0, [100.0])[0]))
    e_inv = max(e_inv, abs(float(b.put_down(88.0)["europea_inv"]) - float(cm.put(100.0, [100.0])[0])))
    return [_fila("Europeas: nota 9 de Kou", "call de Kou contra 9,14732", e_nota, 1e-5),
            _fila("Europeas: FFT contra cuadratura", "error relativo máximo, strikes 70–140", e_fq, 1e-6),
            _fila("Europeas: FFT contra Gil–Pelaez", "error absoluto máximo (puntos, S0=100)", e_gp, 1e-6),
            _fila("Europeas: inversión de F₀ contra Fourier", "call y put en K=100 (puntos, S0=100)", e_inv, 1e-6)]


# ------------------------------------------------------------------- Monte Carlo
REF_MC = dict(S0=100.0, K=100.0, T=0.25, r=0.02, delta=0.015, sigma=0.15, lam=1.5, p=0.3, eta1=25.0, eta2=10.0,
              Lu=115.0, Ld=88.0)


def control_mc(n=10_000_000, seed=0, n_GS=14, B=2, dps=45):
    c = REF_MC
    kq = kou_Q(c["r"], c["delta"], c["sigma"], c["lam"], c["p"], c["eta1"], c["eta2"])
    res = mc.precios_mc(kq, c["S0"], c["K"], c["T"], c["r"], c["delta"], c["Lu"], c["Ld"], n, seed)
    b = Barreras(kq, c["S0"], c["K"], c["T"], c["r"], c["delta"], n_GS, B, dps)
    u, d = b.call_up(c["Lu"]), b.put_down(c["Ld"])
    ref = dict(call=float(u["europea_inv"]), put=float(d["europea_inv"]), uoc=float(u["v_out_inv"]), dop=float(d["v_out_inv"]))
    z = max(abs(res[k][0] - ref[k]) / res[k][1] for k in ref)
    return [_fila("Monte Carlo (control)", f"call, put, up-and-out, down-and-out; {n:g} trayectorias; máx |dif|/error estándar", z, 3.0)], \
        dict(mc=res, inv=ref)
