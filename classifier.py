from regex import classify_with_regex
from bert import classify_with_bert
from llm import classify_with_llm
import pandas as pd

def classify(logs):
    labels = []
    for source, log_msg in logs:
        label = classify_logs(source, log_msg)
        labels.append(label)
    return labels

def classify_logs(source, log_msg):
    if source=='LegacyCRM':
        return classify_with_llm(log_msg)

    else:
        label = classify_with_regex(log_msg)
        if not label:
            label = classify_with_bert(log_msg)

    return label


def classify_csv(input_file):
    df = pd.read_csv(input_file)
    df["target_label"] = classify(list(zip(df["source"], df["log_message"])))
    output_file = "output.csv"
    df.to_csv(output_file, index=False)
    return output_file


if __name__ == '__main__':
    classify_csv("resources/test.csv")





