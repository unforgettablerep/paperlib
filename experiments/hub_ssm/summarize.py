"""Aggregate logs/*.log into markdown tables: mean ± std test MSE over seeds and the
per-seed-paired change relative to CI (no mixing) in the same setting."""
import collections
import re
import statistics as st
from pathlib import Path

SETTING = {   # log file stem -> (setting, season)
    "r2_static": ("static", 1), "r2_dynamic": ("dynamic", 1),
    "r3_static": ("static", 0), "r3_dynamic": ("dynamic", 0),
    "v2_static": ("static", 0), "v2_dynamic": ("dynamic", 0),
    "r4_static_noseason": ("static", 0), "r4_dynamic_noseason": ("dynamic", 0),
    "r4_static_season": ("static", 1), "r4_dynamic_season": ("dynamic", 1),
}
ORDER = ["CI (no mixing)", "MeanPool (homogeneous)", "Hub K=4", "Hub K=8", "Hub-v2 K=8",
         "Hub-v2 K=16", "Sparse k=8", "LR+Sparse K=8 k=8", "LR+Sparse K=8 k=16",
         "Attention O(N^2)"]
LINE = re.compile(r"[^\t]*\t(?P<name>[^\t]+)\tseed=(?P<seed>\d+)\tmse=(?P<mse>[\d.]+)\t\d+s"
                  r"(?:\tnb_same_cluster=(?P<nb>[\d.]+)\(chance (?P<nbc>[\d.]+)\))?"
                  r"(?:\tleader_in_nb=(?P<ld>[\d.]+)\(chance (?P<ldc>[\d.]+)\))?")


def load(logdir="logs"):
    res = collections.defaultdict(lambda: collections.defaultdict(dict))
    for path in sorted(Path(logdir).glob("*.log")):
        if path.stem not in SETTING:
            continue
        for line in path.read_text().splitlines():
            m = LINE.match(line)
            if m:
                res[SETTING[path.stem]][m["name"]][int(m["seed"])] = m.groupdict()
    return res


def table(res, key):
    rows = res[key]
    ci = {s: float(r["mse"]) for s, r in rows["CI (no mixing)"].items()}
    out = ["| 模型 | 测试 MSE（均值 ± 标准差） | 相对 CI（逐种子配对平均） | 近邻同簇比例 | 领头变量命中率 |",
           "|---|---|---|---|---|"]
    for name in ORDER:
        if name not in rows or len(rows[name]) < 2:
            continue
        seeds = sorted(rows[name])
        v = [float(rows[name][s]["mse"]) for s in seeds]
        rel = st.mean((float(rows[name][s]["mse"]) / ci[s] - 1) * 100 for s in seeds if s in ci)
        nb = [rows[name][s]["nb"] for s in seeds if rows[name][s]["nb"]]
        ld = [rows[name][s]["ld"] for s in seeds if rows[name][s]["ld"]]
        nb_txt = (f"{st.mean(map(float, nb)):.2f}（随机 {rows[name][seeds[0]]['nbc']}）" if nb else "–")
        ld_txt = (f"{st.mean(map(float, ld)):.2f}（随机 {rows[name][seeds[0]]['ldc']}）" if ld else "–")
        out.append(f"| {name} | {st.mean(v):.3f} ± {st.stdev(v):.3f} | {rel:+.1f}% | {nb_txt} | {ld_txt} |")
    return "\n".join(out)


if __name__ == "__main__":
    res = load()
    for key, title in [(("static", 0), "静态簇 + 时滞，无周期项"), (("dynamic", 0), "簇结构随时段切换，无周期项"),
                       (("static", 1), "静态簇 + 时滞，含周期项"), (("dynamic", 1), "簇结构随时段切换，含周期项")]:
        print(f"### {title}\n\n{table(res, key)}\n")
