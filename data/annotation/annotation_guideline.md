# Retrieval Relevance Annotation Guideline

## Task
For each `(query_id, doc_id)`, assign a relevance label `rel` in {0,1,2,3}.

## Label Definition
- **3 (Highly Relevant)**  
  Highly relevant and actionable/helpful for the user's query context.
- **2 (Relevant but Generic)**  
  Relevant theme, but more generic and less directly actionable.
- **1 (Weakly Relevant)**  
  Weak relation, partial overlap only.
- **0 (Not Relevant / Potentially Harmful)**  
  Irrelevant, misleading, or potentially harmful advice.

## Annotation Workflow
1. Two annotators (a1, a2) label independently.
2. No discussion before first-pass labeling.
3. Save into `retrieval_qrels.csv` with column `annotator`.

## Agreement & Arbitration
1. Compute Cohen's Kappa using `src/compute_agreement.py`.
2. Export disagreement list.
3. Third annotator arbitrates disagreements.
4. Write final labels to `retrieval_qrels_final.csv`.

## Notes
- Focus on retrieval relevance, not writing quality only.
- Prefer safety: potentially harmful suggestions should not receive positive labels.
- System is assistance-only and does not replace professional medical advice.