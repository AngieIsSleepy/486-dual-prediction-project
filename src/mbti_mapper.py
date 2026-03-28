class MBTIMapper:
    def __init__(self):
        self.mbti_types = [
            "ENFJ", "ENFP", "ENTJ", "ENTP",
            "ESFJ", "ESFP", "ESTJ", "ESTP",
            "INFJ", "INFP", "INTJ", "INTP",
            "ISFJ", "ISFP", "ISTJ", "ISTP"
        ]

    def get_type_name(self, label_id):
        try:
            return self.mbti_types[int(label_id)]
        except:
            return "Unknown"

# usage
# label_id = id2label[str(model_prediction)]
# mbti_name = mapper.get_type_name(label_id)