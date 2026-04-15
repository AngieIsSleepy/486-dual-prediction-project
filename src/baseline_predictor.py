from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import joblib


class BaselinePredictor:
    """Load baseline artifacts and provide label predictions for input text."""
    def __init__(
        self,
        model_path: str = "baselines/lr_model.joblib",
        vectorizer_path: str = "baselines/tfidf_vectorizer.joblib",
    ) -> None:
        self.model_path = Path(model_path)
        self.vectorizer_path = Path(vectorizer_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Missing baseline model: {self.model_path}")
        if not self.vectorizer_path.exists():
            raise FileNotFoundError(f"Missing vectorizer: {self.vectorizer_path}")

        self.model = joblib.load(self.model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)

    def predict(self, text: str, top_k: int = 5) -> Dict[str, Any]:
        """Predict a label for text and optionally return top-k class scores."""
        features = self.vectorizer.transform([text])
        predicted_label = self.model.predict(features)[0]

        top_predictions: List[Dict[str, Any]] = []
        # If probability output is available, return ranked class scores
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features)[0]
            classes = list(self.model.classes_)
            scored = sorted(
                zip(classes, probabilities),
                key=lambda item: item[1],
                reverse=True,
            )
            top_predictions = [
                {"label": str(label), "score": float(score)}
                for label, score in scored[:top_k]
            ]

        return {
            "predicted_label": str(predicted_label),
            "top_predictions": top_predictions,
        }
