"""Aggregate real-data results into markdown tables and paired hypothesis checks.

A change counts as effective on a dataset only if its test MSE is lower than the reference
in every seed (paired by seed) AND the mean paired improvement is at least 1%; lower in
every seed but by less than 1% is reported as a marginal improvement. Model selection
(early stopping) uses the validation split only.
"""
import collections
import re
import statistics as st
import sys

LINE = re.compile(r"(?P<data>\S+)\tT=(?P<T>\d+)\t(?P<model>\S+)\tseed=(?P<seed>\d+)\tval=(?P<val>[\d.]+)"
                  r"\tmse=(?P<mse>[\d.]+)\tmae=(?P<mae>[\d.]+)\tepochs=(?P<ep>\d+)\t(?P<sec>\d+)s")
NAMES = {"none": "CI（无变量混合）", "attn": "注意力（iTransformer 式）", "bimamba": "双向 Mamba（S-Mamba 式）",
         "dyntopk": "动态 top-K（MS-FLOW 式）", "retrieve": "检索后注意（候选来自嵌入）",
         "attn+id": "注意力 + 身份嵌入", "bimamba+id": "双向 Mamba + 身份嵌入",
         "retrieve+id": "检索后注意 + 身份嵌入", "bimamba-shuffled": "双向 Mamba（变量顺序打乱）",
         "bimamba-vpt": "双向 Mamba + 随机顺序训练", "bimamba-vpt+tta4": "双向 Mamba + 随机顺序训练 + 4 顺序平均",
         "bimamba+id-vpt": "双向 Mamba + 身份嵌入 + 随机顺序训练",
         "bimamba+id-vpt+tta4": "双向 Mamba + 身份嵌入 + 随机顺序训练 + 4 顺序平均"}
ORDER = list(NAMES)
CHECKS = [  # (hypothesis, candidate, reference)
    ("H1 稀疏替代注意力", "dyntopk", "attn"), ("H1 稀疏替代注意力", "retrieve", "attn"),
    ("H1 稀疏替代双向 Mamba", "dyntopk", "bimamba"), ("H1 稀疏替代双向 Mamba", "retrieve", "bimamba"),
    ("H2 身份嵌入", "attn+id", "attn"), ("H2 身份嵌入", "bimamba+id", "bimamba"),
    ("H2 身份嵌入", "retrieve+id", "retrieve"),
    ("H4 顺序鲁棒", "bimamba-vpt", "bimamba"), ("H4 顺序鲁棒", "bimamba-vpt+tta4", "bimamba"),
    ("H4 顺序鲁棒", "bimamba-vpt+tta4", "bimamba-vpt"),
    ("H4 顺序鲁棒", "bimamba+id-vpt+tta4", "bimamba+id"),
    ("组合 vs 原版", "bimamba+id-vpt+tta4", "bimamba"), ("组合 vs 原版", "bimamba+id-vpt+tta4", "attn"),
    ("组合 vs 原版", "bimamba+id", "attn"),
    ("参考：变量混合 vs CI", "attn", "none"), ("参考：变量混合 vs CI", "bimamba", "none"),
    ("参考：变量混合 vs CI", "retrieve", "none"),
]


def load(path):
    res = collections.defaultdict(lambda: collections.defaultdict(dict))
    for line in open(path):
        m = LINE.match(line.strip())
        if m:
            res[(m["data"], int(m["T"]))][m["model"]][int(m["seed"])] = {
                k: float(m[k]) for k in ("val", "mse", "mae", "sec", "ep")}
    return res


def fmt(xs):
    return f"{st.mean(xs):.4f} ± {st.stdev(xs):.4f}" if len(xs) > 1 else f"{xs[0]:.4f}"


def tables(res):
    out = []
    for (data, T), rows in sorted(res.items()):
        out += [f"#### {data}（预测 {T} 步）", "",
                "| 模型 | 测试 MSE | 测试 MAE | 种子数 | 平均耗时 |", "|---|---|---|---|---|"]
        for key in ORDER:
            if key in rows:
                r = rows[key]
                out.append(f"| {NAMES[key]} | {fmt([x['mse'] for x in r.values()])} | "
                           f"{fmt([x['mae'] for x in r.values()])} | {len(r)} | "
                           f"{st.mean(x['sec'] for x in r.values()) / 60:.1f} min |")
        out.append("")
    return "\n".join(out)


def checks(res):
    out = ["| 假设 | 数据集 | 候选 vs 参照 | MSE 变化（配对平均） | 更好的种子 | 判定 |",
           "|---|---|---|---|---|---|"]
    for (data, T), rows in sorted(res.items()):
        for hyp, cand, ref in CHECKS:
            if cand not in rows or ref not in rows:
                continue
            seeds = sorted(set(rows[cand]) & set(rows[ref]))
            if len(seeds) < 2:
                continue
            d = [(rows[cand][s]["mse"] / rows[ref][s]["mse"] - 1) * 100 for s in seeds]
            better = sum(x < 0 for x in d)
            if better == len(seeds):
                verdict = "有效" if st.mean(d) <= -1.0 else "微弱改善（<1%）"
            elif better == 0:
                verdict = "变差"
            else:
                verdict = "不确定"
            out.append(f"| {hyp} | {data} | {NAMES[cand]} vs {NAMES[ref]} | {st.mean(d):+.2f}% "
                       f"| {better}/{len(seeds)} | {verdict} |")
    return "\n".join(out)


def order_sensitivity(res):
    out = ["| 数据集 | 原顺序 MSE | 打乱顺序 MSE | 逐种子差异 | 原顺序的种子间标准差 |", "|---|---|---|---|---|"]
    for (data, T), rows in sorted(res.items()):
        a, b = rows.get("bimamba", {}), rows.get("bimamba-shuffled", {})
        seeds = sorted(set(a) & set(b))
        if not seeds:
            continue
        diff = ", ".join(f"{(b[s]['mse'] / a[s]['mse'] - 1) * 100:+.2f}%" for s in seeds)
        sd = st.stdev(x["mse"] for x in a.values()) if len(a) > 1 else float("nan")
        out.append(f"| {data} | {fmt([a[s]['mse'] for s in seeds])} | {fmt([b[s]['mse'] for s in seeds])} "
                   f"| {diff} | {sd:.4f} |")
    return "\n".join(out)


if __name__ == "__main__":
    res = load(sys.argv[1] if len(sys.argv) > 1 else "logs/results_T96.log")
    print(tables(res))
    print("#### 假设检验\n")
    print(checks(res))
    print("\n#### 变量顺序敏感性（双向 Mamba）\n")
    print(order_sensitivity(res))
