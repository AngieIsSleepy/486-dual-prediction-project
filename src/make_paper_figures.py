from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd

try:
    import seaborn as sns
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False


def norm_name(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum())


def find_col(df: pd.DataFrame, aliases: list[str]) -> Optional[str]:
    col_map = {norm_name(c): c for c in df.columns}
    for a in aliases:
        if norm_name(a) in col_map:
            return col_map[norm_name(a)]
    return None


def canonical_system_name(raw: str) -> str:
    s = str(raw).strip().lower()
    if "bm25" in s:
        return "BM25"
    if "dense" in s:
        return "Dense"
    if "rerank" in s:
        return "Reranker"
    return str(raw)


def prepare_ndcg_long(eval_per_query: pd.DataFrame) -> pd.DataFrame:
    q_col = find_col(eval_per_query, ["query_id", "query", "qid"])
    sys_col = find_col(eval_per_query, ["system", "model", "retriever"])
    ndcg_col = find_col(eval_per_query, ["ndcg_at_10", "ndcg@10", "nDCG@10", "ndcg10", "ndcg"])

    if q_col and sys_col and ndcg_col:
        out = eval_per_query[[q_col, sys_col, ndcg_col]].copy()
        out.columns = ["query_id", "system", "ndcg_at_10"]
        out["system"] = out["system"].apply(canonical_system_name)
        out["ndcg_at_10"] = pd.to_numeric(out["ndcg_at_10"], errors="coerce")
        out = out.dropna(subset=["query_id", "system", "ndcg_at_10"])
        return out

    if not q_col:
        raise ValueError("Cannot find query_id column in eval_per_query.csv")

    cols = eval_per_query.columns
    ndcg_candidates: Dict[str, str] = {}
    for c in cols:
        cn = c.lower()
        if "ndcg" not in cn:
            continue
        if "bm25" in cn:
            ndcg_candidates["BM25"] = c
        elif "dense" in cn:
            ndcg_candidates["Dense"] = c
        elif "rerank" in cn:
            ndcg_candidates["Reranker"] = c

    if not ndcg_candidates:
        raise ValueError("Cannot find nDCG@10 columns in eval_per_query.csv")

    rows = []
    for _, r in eval_per_query.iterrows():
        qid = r[q_col]
        for sys_name, c in ndcg_candidates.items():
            v = pd.to_numeric(r[c], errors="coerce")
            if pd.notna(v):
                rows.append({"query_id": qid, "system": sys_name, "ndcg_at_10": float(v)})

    out = pd.DataFrame(rows)
    out = out.dropna(subset=["query_id", "system", "ndcg_at_10"])
    return out


def plot_per_query_ndcg(long_df: pd.DataFrame, out_path: Path) -> None:
    order = [s for s in ["BM25", "Dense", "Reranker"] if s in set(long_df["system"])]
    plt.figure(figsize=(8, 5), dpi=150)

    if HAS_SEABORN:
        sns.violinplot(data=long_df, x="system", y="ndcg_at_10", order=order, inner="box", cut=0)
        sns.stripplot(data=long_df, x="system", y="ndcg_at_10", order=order, color="black", alpha=0.5, size=3)
    else:
        data = [long_df[long_df["system"] == s]["ndcg_at_10"].values for s in order]
        plt.boxplot(data, labels=order)

    plt.title("Per-query nDCG@10 Distribution")
    plt.xlabel("System")
    plt.ylabel("nDCG@10")
    plt.ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def build_delta_table(long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivot = long_df.pivot_table(index="query_id", columns="system", values="ndcg_at_10", aggfunc="mean")
    if "Dense" not in pivot.columns or "Reranker" not in pivot.columns:
        raise ValueError("Need Dense and Reranker nDCG@10 to compute delta.")

    pivot = pivot.dropna(subset=["Dense", "Reranker"]).copy()
    pivot["delta_reranker_minus_dense"] = pivot["Reranker"] - pivot["Dense"]

    eps = 1e-12
    improved = int((pivot["delta_reranker_minus_dense"] > eps).sum())
    unchanged = int((pivot["delta_reranker_minus_dense"].abs() <= eps).sum())
    degraded = int((pivot["delta_reranker_minus_dense"] < -eps).sum())
    mean_delta = float(pivot["delta_reranker_minus_dense"].mean())

    summary = pd.DataFrame(
        [
            {"metric": "improved_queries", "value": improved},
            {"metric": "unchanged_queries", "value": unchanged},
            {"metric": "degraded_queries", "value": degraded},
            {"metric": "mean_delta_ndcg_at_10", "value": mean_delta},
        ]
    )

    per_query = pivot.reset_index()[["query_id", "Dense", "Reranker", "delta_reranker_minus_dense"]]
    return summary, per_query


def plot_delta_hist(delta_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 5), dpi=150)

    vals = delta_df["delta_reranker_minus_dense"].values
    if HAS_SEABORN:
        sns.histplot(vals, bins=20, kde=True)
    else:
        plt.hist(vals, bins=20, alpha=0.8)

    plt.axvline(0.0, color="red", linestyle="--", linewidth=1)
    plt.title("Delta nDCG@10 (Reranker - Dense) per Query")
    plt.xlabel("Delta nDCG@10")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def class_imbalance_plot(cls_df: pd.DataFrame, out_fig: Path, out_table: Path) -> None:
    label_col = find_col(cls_df, ["coarse_label", "label", "class", "target"])
    if not label_col:
        raise ValueError("Cannot find label column in mental_health_classification_clean.csv")

    counts = cls_df[label_col].astype(str).value_counts().rename_axis("label").reset_index(name="count")
    counts.to_csv(out_table, index=False)

    plt.figure(figsize=(10, 5), dpi=150)
    if HAS_SEABORN:
        sns.barplot(data=counts, x="label", y="count")
    else:
        plt.bar(counts["label"], counts["count"])
    plt.title("Class Imbalance in Mental Health Classification Dataset")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_fig)
    plt.close()


def save_acc_macro_table(out_path: Path, acc: Optional[float], macro_f1: Optional[float], metrics_path: Optional[Path]) -> None:
    if metrics_path and metrics_path.exists():
        mdf = pd.read_csv(metrics_path)
        a_col = find_col(mdf, ["accuracy", "acc"])
        f_col = find_col(mdf, ["macro_f1", "macro-f1", "macro f1"])
        if a_col and f_col:
            row = mdf.iloc[0]
            acc = float(row[a_col])
            macro_f1 = float(row[f_col])

    if acc is None or macro_f1 is None:
        return

    out = pd.DataFrame(
        [
            {"metric": "Accuracy", "value": acc},
            {"metric": "Macro-F1", "value": macro_f1},
        ]
    )
    out.to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_per_query", default="artifacts/eval_per_query.csv")
    parser.add_argument("--cls_data", default="data/mental_health_classification_clean.csv")
    parser.add_argument("--cls_metrics", default=None, help="optional CSV containing accuracy/macro_f1")
    parser.add_argument("--acc", type=float, default=None, help="optional accuracy value")
    parser.add_argument("--macro_f1", type=float, default=None, help="optional macro-f1 value")
    parser.add_argument("--out_dir", default="artifacts/paper_figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # per-query nDCG box/violin
    eval_df = pd.read_csv(args.eval_per_query, encoding="utf-8-sig")
    long_df = prepare_ndcg_long(eval_df)
    long_df.to_csv(out_dir / "table_per_query_ndcg_long.csv", index=False)
    plot_per_query_ndcg(long_df, out_dir / "fig_per_query_ndcg_violin_box.png")

    # delta histogram + summary table
    delta_summary, delta_per_query = build_delta_table(long_df)
    delta_summary.to_csv(out_dir / "table_delta_summary.csv", index=False)
    delta_per_query.to_csv(out_dir / "table_delta_per_query.csv", index=False)
    plot_delta_hist(delta_per_query, out_dir / "fig_delta_ndcg_hist.png")


if __name__ == "__main__":
    main()