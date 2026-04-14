from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels", default="data/annotations/retrieval_qrels.csv")
    parser.add_argument("--output_json", default="artifacts/annotation_agreement.json")
    parser.add_argument("--output_disagreements", default="data/annotations/retrieval_disagreements.csv")
    args = parser.parse_args()

    qrels_path = Path(args.qrels)
    output_json_path = Path(args.output_json)
    output_disagreements_path = Path(args.output_disagreements)

    df = pd.read_csv(qrels_path)
    required_cols = {"query_id", "doc_id", "rel", "annotator"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"qrels must contain columns: {required_cols}")

    df["rel"] = df["rel"].astype(int)

    pivot = (
        df.pivot_table(
            index=["query_id", "doc_id"],
            columns="annotator",
            values="rel",
            aggfunc="first",
        )
        .reset_index()
    )

    annotators = sorted([c for c in pivot.columns if c not in ("query_id", "doc_id")])
    if len(annotators) < 2:
        raise ValueError("Need at least two annotators to compute agreement.")

    a_name, b_name = ("a1", "a2") if {"a1", "a2"}.issubset(set(annotators)) else (annotators[0], annotators[1])

    paired = pivot.dropna(subset=[a_name, b_name]).copy()
    paired[a_name] = paired[a_name].astype(int)
    paired[b_name] = paired[b_name].astype(int)

    if paired.empty:
        raise ValueError("No paired samples found between the two annotators.")

    a = paired[a_name].tolist()
    b = paired[b_name].tolist()

    kappa = cohen_kappa_score(a, b)
    kappa_linear = cohen_kappa_score(a, b, weights="linear")
    kappa_quadratic = cohen_kappa_score(a, b, weights="quadratic")
    exact_agreement = float((paired[a_name] == paired[b_name]).mean())

    confusion = pd.crosstab(paired[a_name], paired[b_name], dropna=False)

    disagreements = paired[paired[a_name] != paired[b_name]].copy()
    disagreements["abs_diff"] = (disagreements[a_name] - disagreements[b_name]).abs()
    disagreements = disagreements.sort_values(by=["abs_diff", "query_id", "doc_id"], ascending=[False, True, True])

    output_disagreements_path.parent.mkdir(parents=True, exist_ok=True)
    disagreements.to_csv(output_disagreements_path, index=False)

    metrics = {
        "num_total_rows": int(len(df)),
        "num_paired_rows": int(len(paired)),
        "annotator_a": a_name,
        "annotator_b": b_name,
        "exact_agreement_rate": exact_agreement,
        "cohen_kappa": float(kappa),
        "cohen_kappa_linear_weighted": float(kappa_linear),
        "cohen_kappa_quadratic_weighted": float(kappa_quadratic),
        "num_disagreements": int(len(disagreements)),
        "confusion_matrix": confusion.to_dict(),
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Disagreements saved to: {output_disagreements_path}")


if __name__ == "__main__":
    main()