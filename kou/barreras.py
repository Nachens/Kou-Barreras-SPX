"""Precios de barreras y probabilidades de cruce bajo Kou (Obs. 4.8, 4.9 y §5.4.1 de la tesis).

Motor: Prop. 4.7 con la resolvente explícita + Gaver-Stehfest (kou.inversion).
Las raíces dependen de (parámetros, alpha) y no de a ni de b: se calculan una vez por parámetros y
se reusan para toda la grilla de barreras.

Inversión adaptativa: con `adaptativo`, cada probabilidad se
invierte con n = n0, n0 + 1, ... hasta que dos diferencias seguidas |f*_n - f*_(n-1)| quedan por debajo de `tol`
(o se llega a n_max, y entonces se marca como no convergida). La precisión sube por tramos de n
(kou.inversion.TRAMOS_N): dps = dps_defecto(extremo del tramo). Las transformadas en cada alpha_j se evalúan una
vez por probabilidad y sirven para todos los n. Con n fijo (adaptativo = False) la inversión usa ese n en todas las probabilidades.
"""
from __future__ import annotations

import mpmath as mp

from . import transformadas as tf
from .inversion import dps_defecto, indices, invertir, invertir_desde, pesos_gs, tramo
from .modelo import Kou


class Resolvente:
    """Raíces cacheadas de G(beta) = j ln2/T para un juego de parámetros."""

    def __init__(self, kou: Kou, T, n=14, B=2, dps=None, adaptativo=False, tol=1e-6, n_max=100):
        self.n, self.B, self.T = n, B, mp.mpf(T)
        self.adaptativo, self.tol, self.n_max = adaptativo, tol, n_max
        self._kou_orig = kou
        self.dps = dps or dps_defecto(n, B)
        if adaptativo:
            self.dps = max(self.dps, dps_defecto(tramo(n), B))
        with mp.workdps(self.dps):
            self.kou = kou.to_mp()
        self._rs = {}
        self.alpha = {}
        self.ultimo = dict(n=n, convergio=True)            # n usado y convergencia de la última inversión

    def _subir_dps(self, dps):
        """Más dígitos: se recalculan los parámetros y se vacía la caché de raíces."""
        self.dps = dps
        with mp.workdps(dps):
            self.kou = self._kou_orig.to_mp()
        self._rs, self.alpha = {}, {}

    def raices(self, j):
        if j not in self._rs:
            from .raices import raices
            with mp.workdps(self.dps):
                a = j * mp.log(2) / self.T
                self.alpha[j] = a
                self._rs[j] = raices(self.kou, a)
        return self._rs[j]

    def invertir(self, g):
        """Invierte la transformada alpha -> g(rs, alpha) (con rs = raíces en alpha)."""
        if self.adaptativo:
            return self._invertir_adaptativo(g)
        with mp.workdps(self.dps):
            def fhat(j):
                rs = self.raices(j)
                return g(rs, self.alpha[j])
            self.ultimo = dict(n=self.n, convergio=True)
            return invertir(fhat, self.T, self.n, self.B)

    def _invertir_adaptativo(self, g):
        n, B = self.n, self.B
        difs, prev, fv = [], None, {}
        while True:
            need = dps_defecto(tramo(n), B)
            if need > self.dps:                       # nuevo tramo: más dígitos, transformadas de nuevo
                self._subir_dps(need)
                fv, prev, difs = {}, None, []
            with mp.workdps(self.dps):
                for j in pesos_gs(n, B):
                    if j not in fv:
                        fv[j] = g(self.raices(j), self.alpha[j])
                v = invertir_desde(fv, self.T, n, B)
            if prev is not None:
                difs.append(abs(v - prev))
                if len(difs) >= 2 and difs[-1] < self.tol and difs[-2] < self.tol:
                    self.ultimo = dict(n=n, convergio=True)
                    return v
            if n >= self.n_max:
                self.ultimo = dict(n=n, convergio=False)
                return v
            prev = v
            n += 1

    # ---- magnitudes básicas ----
    def prob_joint(self, a, b):
        """P(X_T >= a, tau_b <= T)."""
        return self.invertir(lambda rs, al: tf.lt_conjunta(a, b, self.kou, rs)[0])

    def prob_marg(self, y):
        """P(X_T >= y), por inversión de F0 (control independiente de Fourier)."""
        return self.invertir(lambda rs, al: tf.lt_marginal(y, self.kou, rs))

    def prob_cruce(self, b):
        """(P(tau_b <= T, Z=0), P(tau_b <= T, Z>0))."""
        pc = self.invertir(lambda rs, al: tf.lt_cruce(b, self.kou, rs, al)[0])
        u1 = self.ultimo
        pj = self.invertir(lambda rs, al: tf.lt_cruce(b, self.kou, rs, al)[1])
        self.ultimo = _peor(u1, self.ultimo)
        return pc, pj


def _peor(*us):
    """El mayor n usado y si todas convergieron."""
    return dict(n=max(u["n"] for u in us), convergio=all(u["convergio"] for u in us))


class Barreras:
    """Precios knock-out/knock-in de una call up-and-out y una put down-and-out bajo Q.

    kouQ: parámetros bajo Q (gamma con G^Q(1) = r - delta). Exige eta1 > 1.
    """

    def __init__(self, kouQ: Kou, S0, K, T, r, delta, n=14, B=2, dps=None, adaptativo=False, tol=1e-6, n_max=100):
        self.S0, self.K, self.T, self.r, self.delta = S0, K, T, r, delta
        self.n, self.B = n, B
        self.dps = dps or dps_defecto(n, B)
        op = dict(adaptativo=adaptativo, tol=tol, n_max=n_max)
        # los parámetros transformados se calculan con mucha precisión: cada resolvente los vuelve a convertir
        # a su propia precisión cuando la sube
        with mp.workdps(max(self.dps, 300)):
            kq = kouQ.to_mp()
            kn = kq.negar()
            kq1, kn1 = kq.esscher(1), kn.esscher(-1)
        self.rQ = Resolvente(kq, T, n, B, self.dps, **op)
        self.rQ1 = Resolvente(kq1, T, n, B, self.dps, **op)
        self.rN = Resolvente(kn, T, n, B, self.dps, **op)
        self.rN1 = Resolvente(kn1, T, n, B, self.dps, **op)

    def fijar_n(self, n):
        """Cambia n de Gaver-Stehfest reusando las raíces ya calculadas (para el estudio de convergencia)."""
        self.n = n
        for r_ in (self.rQ, self.rQ1, self.rN, self.rN1):
            r_.n = n

    # ------------------------------------------------------------ call up-and-out
    def call_up(self, L):
        """Dict con V_in, la call por inversión de F0 y la ccruce, para la call up-and-out.

        Devuelve 0 en todo si K >= L (la knock-out vale 0). b = log(L/S0) > 0.
        """
        S0, K, T, r, d = self.S0, self.K, self.T, self.r, self.delta
        if K >= L:
            z = mp.mpf(0)
            return dict(v_in=z, europea_inv=z, v_out_inv=z, trivial=True, n_usado=0, convergio=True)
        with mp.workdps(self.dps):
            k = mp.log(mp.mpf(K) / S0)
            b = mp.log(mp.mpf(L) / S0)
            jQ = self.rQ.prob_joint(k, b)
            u = [self.rQ.ultimo]
            j1 = self.rQ1.prob_joint(k, b)
            u.append(self.rQ1.ultimo)
            mQ = self.rQ.prob_marg(k)
            u.append(self.rQ.ultimo)
            m1 = self.rQ1.prob_marg(k)
            u.append(self.rQ1.ultimo)
            dfd, dfr = mp.exp(-d * T), mp.exp(-r * T)
            v_in = S0 * dfd * j1 - K * dfr * jQ
            call = S0 * dfd * m1 - K * dfr * mQ
            pu = _peor(*u)
            return dict(v_in=v_in, europea_inv=call, v_out_inv=call - v_in, trivial=False, n_usado=pu["n"],
                        convergio=pu["convergio"])

    # ----------------------------------------------------------- put down-and-out
    def put_down(self, L):
        """Análogo para la put down-and-out: X' = -X, b' = log(S0/L), k' = log(S0/K).

        Devuelve 0 si K <= L.
        """
        S0, K, T, r, d = self.S0, self.K, self.T, self.r, self.delta
        if K <= L:
            z = mp.mpf(0)
            return dict(v_in=z, europea_inv=z, v_out_inv=z, trivial=True, n_usado=0, convergio=True)
        with mp.workdps(self.dps):
            kp = mp.log(mp.mpf(S0) / K)
            bp = mp.log(mp.mpf(S0) / L)
            jQ = self.rN.prob_joint(kp, bp)
            u = [self.rN.ultimo]
            j1 = self.rN1.prob_joint(kp, bp)
            u.append(self.rN1.ultimo)
            mQ = self.rN.prob_marg(kp)
            u.append(self.rN.ultimo)
            m1 = self.rN1.prob_marg(kp)
            u.append(self.rN1.ultimo)
            dfd, dfr = mp.exp(-d * T), mp.exp(-r * T)
            v_in = K * dfr * jQ - S0 * dfd * j1
            put = K * dfr * mQ - S0 * dfd * m1
            pu = _peor(*u)
            return dict(v_in=v_in, europea_inv=put, v_out_inv=put - v_in, trivial=False, n_usado=pu["n"],
                        convergio=pu["convergio"])

    # ---------------------------------------------------------------- cruce
    def cruce(self, L, tipo):
        """(Q(tau<=T, Z=0), Q(tau<=T, Z>0), fracción por salto). tipo: 'call_up' o 'put_down'."""
        R = self.rQ if tipo == "call_up" else self.rN
        with mp.workdps(max(R.dps, 300)):
            b = mp.log(mp.mpf(L) / self.S0) if tipo == "call_up" else mp.log(mp.mpf(self.S0) / L)
        pc, pj = R.prob_cruce(b)
        self.ultimo_cruce = R.ultimo
        with mp.workdps(R.dps):
            tot = pc + pj
            return pc, pj, (pj / tot if tot != 0 else mp.mpf(0))
