"""Create run_quantum2.py from run_quantum.py: model selection on the validation split (not test), outputs quantum2_<name>.json."""
import pathlib
src = pathlib.Path(__file__).resolve().parent / "run_quantum.py"
s = src.read_text(encoding="utf-8")
a = s.index('    preds = {"M0": te["p_M0"].to_numpy()}')
b = s.index('    out = {"name": name,')
new = '''    preds, vpred = {"M0": te["p_M0"].to_numpy()}, {}
    def put(k, model, Xv_, Xt_):
        vpred[k] = model.predict_proba(Xv_)[:, 1]; preds[k] = model.predict_proba(Xt_)[:, 1]
    put("LR_full", LogisticRegression(C=0.5, max_iter=3000).fit(Xf, yf), Xv, Xt)
    put("LR_d", LogisticRegression(C=0.5, max_iter=3000).fit(Af, yf), Av, At)
    put("LR_d_strong", LogisticRegression(C=0.01, max_iter=3000).fit(Af, yf), Av, At)
    put("HGB_d", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=SEED).fit(Af, yf), Av, At)
    fm = Q.fit_fourier(Af, yf); vpred["CQ1_fourier"] = fm.predict_proba(Q.fourier_features(Av))[:, 1]; preds["CQ1_fourier"] = fm.predict_proba(Q.fourier_features(At))[:, 1]
    timing = {}
    for L in (2, 4):
        s0 = time.time()
        m = Q.fit_vqc(Af, yf, Av, yv, d, layers=L, seed=SEED)   # Av is also used for early stopping of the VQC
        vpred[f"Q1_vqc_L{L}"] = Q.predict_vqc(m, Av); preds[f"Q1_vqc_L{L}"] = Q.predict_vqc(m, At); timing[f"Q1_L{L}"] = time.time() - s0
    ks = np.sort(rng.choice(len(Af), min(N_KERNEL, len(Af)), replace=False))
    Ak, yk = Af[ks], yf[ks]
    if len(np.unique(yk)) == 2:
        Sk, Sv, St = Q.zz_states(Ak), Q.zz_states(Av), Q.zz_states(At)
        svc = Q.fit_kernel_svc(Q.fidelity_kernel(Sk, Sk), yk, SEED)
        vpred["Q2_fidelity_kernel"] = svc.predict_proba(Q.fidelity_kernel(Sv, Sk))[:, 1]; preds["Q2_fidelity_kernel"] = svc.predict_proba(Q.fidelity_kernel(St, Sk))[:, 1]
        Bk, Bv, Bt = Q.bloch_vectors(Sk, d), Q.bloch_vectors(Sv, d), Q.bloch_vectors(St, d)
        g = 1.0 / (Bk.var() * Bk.shape[1] + 1e-9)
        svc = SVC(kernel="precomputed", probability=True, random_state=SEED).fit(Q.rbf(Bk, Bk, g), yk)
        vpred["Q3_projected_kernel"] = svc.predict_proba(Q.rbf(Bv, Bk, g))[:, 1]; preds["Q3_projected_kernel"] = svc.predict_proba(Q.rbf(Bt, Bk, g))[:, 1]
        g2 = 1.0 / (Ak.var() * Ak.shape[1] + 1e-9)
        svc = SVC(kernel="precomputed", probability=True, random_state=SEED).fit(Q.rbf(Ak, Ak, g2), yk)
        vpred["CQ2_rbf_svm"] = svc.predict_proba(Q.rbf(Av, Ak, g2))[:, 1]; preds["CQ2_rbf_svm"] = svc.predict_proba(Q.rbf(At, Ak, g2))[:, 1]
    groups = {k: np.asarray(v) for k, v in te.reset_index(drop=True).groupby("entity").indices.items()}
    res = {}
    for k, p in preds.items():
        res[k] = core.skill(yt, p, preds["M0"]); res[k]["auc"] = core.auc(yt, p)
        if k in vpred:
            res[k]["val_ll"] = core.log_loss(yv, vpred[k])
    qn = [k for k in vpred if k.startswith("Q")]
    cn_prereg = [k for k in ["LR_d", "HGB_d", "CQ1_fourier", "CQ2_rbf_svm"] if k in vpred]
    bq = min(qn, key=lambda k: res[k]["val_ll"]); bcp = min(cn_prereg, key=lambda k: res[k]["val_ll"])   # selected on validation
    bc = bcp
    ll = lambda p: -(yt * np.log(np.clip(p, 1e-6, 1)) + (1 - yt) * np.log(np.clip(1 - p, 1e-6, 1)))
    lq, lc = ll(preds[bq]), ll(preds[bcp]); lcp = lc
    dm = core.diebold_mariano(lq, lc); dm_prereg = dm
    bs = core.cluster_bootstrap(groups, lambda idx: lq[idx].mean() - lc[idx].mean(), reps=2000)
    from scipy import stats
    p_q_better = float(stats.norm.cdf(dm["dm"]))
    p_boot_q_better = float(np.mean(bs >= 0))  # rev2 fix: one-sided p for "quantum better" (ll diff Q-C < 0)
'''
s = s[:a] + new + s[b:]
s = s.replace('"p_quantum_better_one_sided": p_q_better,', '"p_quantum_better_one_sided": p_q_better, "p_boot_quantum_better": p_boot_q_better, "selection": "validation log loss",')
s = s.replace('(ROOT / "results" / f"quantum_{name}.json")', '(ROOT / "results" / f"quantum2_{name}.json")')
(src.parent / "run_quantum2.py").write_text(s, encoding="utf-8")
print("written", "quantum2_" in s, "val_ll" in s)
