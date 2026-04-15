import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Any, Dict, List


class QueryAnalyzer:
    """Analyze user text and return top mental-state category predictions."""
    def __init__(self, mental_model_path: str = "models/mental_state_distilbert"):
        # Load tokenizer and fine-tuned mental-state classification model
        self.mental_tokenizer = AutoTokenizer.from_pretrained(mental_model_path)
        self.mental_model = AutoModelForSequenceClassification.from_pretrained(mental_model_path)

    def analyze(self, user_text: str, top_k: int = 5) -> Dict[str, Any]:
        """Run inference on user text and return top-k labels with scores."""
        results: Dict[str, Any] = {}
        # Tokenize input text for model inference
        inputs = self.mental_tokenizer(
            user_text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        # Disable gradients for efficient inference
        with torch.no_grad():
            outputs = self.mental_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
            top_probs, top_indices = torch.topk(probs, min(top_k, probs.shape[-1]))
         # Build ranked label list and a label->score map
        top_list: List[Dict[str, Any]] = []
        categories: Dict[str, float] = {}
        for i in range(len(top_indices)):
            label_id = int(top_indices[i].item())
            label_name = self.mental_model.config.id2label[label_id]
            score = float(top_probs[i].item())
            top_list.append({"label": str(label_name), "score": score})
            categories[str(label_name)] = score

        results["mental_state_top5"] = top_list
        results["categories"] = categories
        return results