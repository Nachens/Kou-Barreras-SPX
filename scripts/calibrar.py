"""Calibración (Kou multi-start y Black-Scholes), SVD, perfil en lambda, conjunto C y Tabla 5.5."""
import json
import time

import numpy as np

from _comun import C, cargar_cfg, escribir_tabla, parser, preparar, salida
from calib import identificabilidad as ident
from calib.objetivo import NOMBRES, theta_a_u
from calib.optimizar import calibrar_bs, calibrar_kou

if __name__ == "__main__":
    a = parser("calibración").parse_args()
    cfg = cargar_cfg(a.config[0])
    cal_cfg = cfg["calibracion"]
    t0 = time.time()
    cad, df, info, prob = preparar(cfg)
    n_ini = cal_cfg["n_inicios"]
    print(f"{cfg['nombre']}: {len(df)} opciones, r={info['r']:.5f}, delta={info['delta']:.5f}, eps0^2={prob.eps0_sq:.4g}")
    k = calibrar_kou(prob, n_ini, cal_cfg["factor_parecido"], cal_cfg["semilla"], jobs=a.jobs)
    print(f"  Kou: R_min={k['R_min']:.4g}, theta={np.round(k['optimo_theta'], 4)}, {k['n_distintos']} mínimos distintos, "
          f"{len(k['minimos'])} conservados")
    for nom, cnt in k["minimos_cerca_de_cota"].items():
        if cnt:
            print(f"  AVISO: {cnt} de los {len(k['minimos'])} mínimos conservados tienen {nom} a menos de 1 % de una cota")
    bs = calibrar_bs(prob)
    sig_bs = bs["sigma_LS"] if cal_cfg["sigma_bs"] == "LS" else bs["sigma_ATM"]
    th_opt = k["optimo_theta"]

    # métricas por modelo
    m_kou = [prob.metricas(prob.precios_modelo(np.array(m["theta"]), cal_cfg["N_fft_final"])) for m in k["minimos"]]
    m_bs = prob.metricas(prob.precios_bs(sig_bs))

    # identificabilidad
    svd = ident.svd_jacobiano(prob, k["optimo_u"])
    pl = cfg["perfil_lambda"]
    perfil = ident.perfil_lambda(prob, k["optimo_u"], pl["n"], pl["lam_min"], pl["lam_max"], pl["reinicios"], cal_cfg["semilla"],
                                 lam_extra=[m["theta"][1] for m in k["minimos"]])
    cand = [dict(theta=m["theta"], R=m["R"], origen="minimo") for m in k["minimos"]]
    cand += [dict(theta=p["theta"], R=p["R"], origen="perfil", lam=p["lam"]) for p in perfil]
    en_C = ident.conjunto_C(prob, cand, k["R_min"], cal_cfg["criterio_C"], cal_cfg["kappa"])
    print(f"  criterio C={cal_cfg['criterio_C']}: {len(en_C)} de {len(cand)} candidatos en C", "(C VACÍO)" if not en_C else "")
    respaldo = None
    if not en_C and cal_cfg.get("criterio_C_respaldo"):
        crit_r = cal_cfg["criterio_C_respaldo"]
        en_C_r = ident.conjunto_C(prob, cand, k["R_min"], crit_r, cal_cfg["kappa"])
        respaldo = dict(criterio=crit_r, kappa=cal_cfg["kappa"], puntos=en_C_r, vacio=len(en_C_r) == 0)
        print(f"  criterio de respaldo {crit_r} (kappa={cal_cfg['kappa']}): {len(en_C_r)} de {len(cand)} candidatos")
    th_opt_en_C = ident.en_C(prob, k["optimo_theta"], k["R_min"], cal_cfg["criterio_C"], cal_cfg["kappa"], k["R_min"])

    # cantidades de contexto para las barreras
    F = cad.S0 * np.exp((info["r"] - info["delta"]) * cad.T)
    i_atm = bs["i_atm"]
    s_bar = float(np.mean(prob.ask - prob.bid))
    s_atm = float(prob.ask[i_atm] - prob.bid[i_atm])
    out = dict(
        nombre=cfg["nombre"], S0=cad.S0, T=cad.T, r=info["r"], delta=info["delta"], F=float(F), n_opciones=len(df),
        limpieza=info, eps0_sq=prob.eps0_sq, n_sin_iv=prob.n_sin_iv, s_bar=s_bar, s_atm=s_atm, K_atm=float(prob.K[i_atm]),
        sigma_atm_imp=float(info["sigma_atm"]), sigma_atm_final=float(prob.iv_mkt[i_atm]),
        K_min=float(prob.K.min()), K_max=float(prob.K.max()), cotas=cfg["cotas"],
        bs=dict(**bs, sigma_elegida=sig_bs, metricas=m_bs),
        kou=dict(optimo=dict(theta=list(map(float, th_opt)), R=k["R_min"], u=list(map(float, k["optimo_u"]))),
                 minimos=k["minimos"], metricas_minimos=m_kou, n_distintos=k["n_distintos"], n_inicios=n_ini,
                 minimos_cerca_de_cota=k["minimos_cerca_de_cota"],
                 R_todos=k["R_todos"]),
        svd=svd, perfil=perfil, C=dict(criterio=cal_cfg["criterio_C"], kappa=cal_cfg["kappa"], puntos=en_C,
                                       vacio=len(en_C) == 0, n_candidatos=len(cand), respaldo=respaldo,
                                       theta_opt_en_C=bool(th_opt_en_C)),
        nombres_theta=NOMBRES)
    d = salida(cfg)
    (d / "calibracion.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    par = dict(info["paridad"], S0=cad.S0, T=cad.T, tasa_externa=info.get("tasa_externa"))
    (d / "paridad.json").write_text(json.dumps(par, indent=2, default=float), encoding="utf-8")

    # Tabla 5.5
    filas = [dict(fecha=cfg["nombre"], modelo="BS", sigma=sig_bs, lam=None, p=None, eta1=None, eta2=None,
                  rmse=m_bs["rmse_vol"], maxv=m_bs["max_vol"], pct=100 * m_bs["frac_en_bidask"])]
    for i, (m, mm) in enumerate(zip(k["minimos"], m_kou)):
        s, l, p_, e1, e2 = m["theta"]
        filas.append(dict(fecha=cfg["nombre"], modelo="Kou" if i == 0 else f"Kou ({i + 1})", sigma=s, lam=l, p=p_, eta1=e1,
                          eta2=e2, rmse=mm["rmse_vol"], maxv=mm["max_vol"], pct=100 * mm["frac_en_bidask"]))
    escribir_tabla("tabla_5_5_parametros", filas, ["fecha", "modelo", "sigma", "lam", "p", "eta1", "eta2", "rmse", "maxv", "pct"], d,
                   tex_cols=["Fecha", "Modelo", "$\\sigma$", "$\\lambda$", "$p$", "$\\eta_1$", "$\\eta_2$", "RMSE vol", "Máx.\\ vol", "\\% en bid--ask"],
                   spec="llrrrrrrrr", fmt=dict(sigma="{:.4f}", lam="{:.3f}", p="{:.3f}", eta1="{:.2f}", eta2="{:.2f}",
                                              rmse="{:.5f}", maxv="{:.5f}", pct="{:.1f}"))
    C.escribir_resumen(cfg, t0, extra=dict(n_en_C=len(en_C)), nombre="resumen_calibrar.json")
    print(f"  listo en {time.time() - t0:.0f} s")
