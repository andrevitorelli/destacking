"""Write the gamma_inference.py configs of every paper 1.1 simulation chain.

Each of the 132 chains in chains.csv (84 DES-like, 48 LSST-like realisations; injected
gamma_cm = k/36, k = 0..6) gets a simulate and a joint config, filled from the
templates in templates/ with the chain's injected gamma and random-seed offset.  It also writes the
simulate configs of the 24 clean DES-like realisations of Appendix D (DESclean0..23).

usage: python make_configs.py [OUT_DIR]      (default $DESTACKING_DATA/output/v1.1/configs)
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from paths import OUT, V11  # noqa: E402


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(V11, "configs")
    os.makedirs(out_dir, exist_ok=True)
    rows = list(csv.DictReader(open(os.path.join(HERE, "chains.csv"))))
    for r in rows:
        tag = "DES" if r["configuration"] == "DES-like" else "LSST"
        fill = {"{GAMMA}": f"{float(r['gamma_true']):.6f}", "{CHAIN}": r["chain"], "{SEED}": r["seed_offset"],
                "{DATA_FOLDER}": os.path.join(OUT, f"{r['chain']}_sim_analytic")}
        for stage in ("simulate", "joint"):
            txt = open(os.path.join(HERE, "templates", f"{tag}_{stage}.yaml")).read()
            for k, v in fill.items():
                txt = txt.replace(k, v)
            open(os.path.join(out_dir, f"{r['chain']}_{stage}.yaml"), "w").write(txt)
    seeds = {r["chain"]: r["seed_offset"] for r in rows}
    # Section 4 LSST-like slope trace: 12 extra realisations at k = 5, 6 (seed offsets k*1000 + r)
    extra = []
    for k in (5, 6):
        for r in range(7, 13):
            name = f"LSSTind{r}k{k}L"; seeds[name] = str(k * 1000 + r)
            extra.append({"chain": name, "configuration": "LSST-like", "k": k, "seed_offset": seeds[name], "variant": ""})
    # Section 4 controls, same seeds as the parent chain: miscentering off (M0) and triaxiality off (T0)
    for k, rs in ((2, range(1, 7)), (3, range(1, 7)), (5, range(1, 13)), (6, range(1, 13))):
        for r in rs:
            extra.append({"chain": f"LSSTind{r}k{k}M0L", "configuration": "LSST-like", "k": k,
                          "seed_offset": seeds[f"LSSTind{r}k{k}L"], "variant": "miscentering: false"})
    for r in range(1, 13):
        extra.append({"chain": f"LSSTind{r}k5T0L", "configuration": "LSST-like", "k": 5,
                      "seed_offset": seeds[f"LSSTind{r}k5L"], "variant": "triaxiality: false"})
    for e in extra:
        fill = {"{GAMMA}": f"{e['k'] / 36:.6f}", "{CHAIN}": e["chain"], "{SEED}": e["seed_offset"],
                "{DATA_FOLDER}": os.path.join(OUT, f"{e['chain']}_sim_analytic")}
        for stage in ("simulate", "joint"):
            txt = open(os.path.join(HERE, "templates", f"LSST_{stage}.yaml")).read()
            for kk, v in fill.items():
                txt = txt.replace(kk, v)
            if stage == "simulate" and e["variant"]:
                key = e["variant"].split(":")[0]
                txt = txt.replace(f"{key}: true", e["variant"])
            open(os.path.join(out_dir, f"{e['chain']}_{stage}.yaml"), "w").write(txt)
    # Appendix E: joint fits with the galaxy-bootstrap covariance, Gaussian (jointB) and per-bin
    # Sellentin-Heavens (jointS) likelihoods, for the 132 scan chains and the 12 LSST-like trace chains
    for name in [r["chain"] for r in rows] + [e["chain"] for e in extra if not e["variant"]]:
        txt = open(os.path.join(out_dir, f"{name}_joint.yaml")).read()
        txt = txt.replace(f"{name}_sim_analytic", f"{name}_sim")
        open(os.path.join(out_dir, f"{name}_jointB.yaml"), "w").write(txt)
        open(os.path.join(out_dir, f"{name}_jointS.yaml"), "w").write(txt + "likelihood: sellentin_heavens\n")
    # Appendix D: 24 clean DES-like realisations (simulate stage only)
    for i in range(24):
        txt = open(os.path.join(HERE, "templates", "DESclean_simulate.yaml")).read()
        txt = txt.replace("{CHAIN}", f"DESclean{i}").replace("{SEED}", str(2000 + i))
        open(os.path.join(out_dir, f"DESclean{i}_simulate.yaml"), "w").write(txt)
    print(f"wrote configs for {len(rows)} scan chains, {len(extra)} Section 4 extra/control chains, "
          f"the Appendix E bootstrap-covariance refits and 24 Appendix D realisations in {out_dir}")


if __name__ == "__main__":
    main()
