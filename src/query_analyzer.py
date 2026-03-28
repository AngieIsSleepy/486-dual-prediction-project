import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class QueryAnalyzer:
    def __init__(self, mental_model_path="models/mental_state_distilbert", mbti_model_path="models/personality_distilbert"):
        print("Loading Model...")
        # Mental State
        self.mental_tokenizer = AutoTokenizer.from_pretrained(mental_model_path)
        self.mental_model = AutoModelForSequenceClassification.from_pretrained(mental_model_path)
        
        # MBTI
        self.mbti_tokenizer = AutoTokenizer.from_pretrained(mbti_model_path)
        self.mbti_model = AutoModelForSequenceClassification.from_pretrained(mbti_model_path)
        
        self.mbti_labels = [
            "ENFJ", "ENFP", "ENTJ", "ENTP", "ESFJ", "ESFP", "ESTJ", "ESTP",
            "INFJ", "INFP", "INTJ", "INTP", "ISFJ", "ISFP", "ISTJ", "ISTP"
        ]

    def analyze(self, user_text):
        """
        Input: user texts
        Output: five mental issues with probabilities and a personality
        """
        results = {}
        
        mental_inputs = self.mental_tokenizer(user_text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            mental_outputs = self.mental_model(**mental_inputs)
            probs = torch.nn.functional.softmax(mental_outputs.logits, dim=-1)[0]
            top5_probs, top5_indices = torch.topk(probs, 5)
            
            top5_list = []
            for i in range(5):
                label_id = top5_indices[i].item()
                label_name = self.mental_model.config.id2label[label_id]
                score = top5_probs[i].item()
                top5_list.append({"label": label_name, "score": score})
            results['mental_state_top5'] = top5_list

        mbti_inputs = self.mbti_tokenizer(user_text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            mbti_outputs = self.mbti_model(**mbti_inputs)
            mbti_probs = torch.nn.functional.softmax(mbti_outputs.logits, dim=-1)[0]
            top5_mbti_probs, top5_mbti_indices = torch.topk(mbti_probs, 5)
            
            mbti_list = []
            for i in range(5):
                pred_id = top5_mbti_indices[i].item()
                dataset_label_idx = self.mbti_model.config.id2label[pred_id]
                mbti_name = self.mbti_labels[int(dataset_label_idx)]
                score = top5_mbti_probs[i].item()
                mbti_list.append({"label": mbti_name, "score": float(score)})
            results['mbti_top5'] = mbti_list

        return results

# For member D usage:
# analyzer = QueryAnalyzer()
# print(analyzer.analyze("I am so stressed about my final exams and feel totally lost."))