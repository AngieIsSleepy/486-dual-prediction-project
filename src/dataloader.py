import pandas as pd
import os

class MentalHealthDataLoader:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir

    def load_classification_data(self):
        """Returns the cleaned Reddit data for Member B's training"""
        path = os.path.join(self.data_dir, "mental_health_classification_clean.csv")
        return pd.read_csv(path)

    def load_qa_corpus(self):
        """Returns the QA corpus for Member C and D's retrieval"""
        path = os.path.join(self.data_dir, "qa_corpus_clean.csv")
        return pd.read_csv(path)

    def load_mbti_data(self):
        """Returns the MBTI posts for personality prediction"""
        path = os.path.join(self.data_dir, "mbti_clean.csv")
        return pd.read_csv(path)

# 使用示例：
# loader = MentalHealthDataLoader()
# df = loader.load_classification_data()