!pip install evaluate rouge_score bert_score datasets transformers
import os
import time
import torch
import numpy as np
import pandas as pd
import json
import re
import string
import matplotlib.pyplot as plt
import unicodedata
import math
from datasets import Dataset, Features, Value, Sequence
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
    pipeline
)
import evaluate
from google.colab import drive


drive.mount('/content/drive') # access Google Drive



FEATURES = Features({
    "id": Value("string"),
    "context": Value("string"),
    "question": Value("string"),
    "answers": Sequence({
        "text": Value("string"),
        "answer_start": Value("int32"),
    }),
})

def robust_gen(data_path):
    """ This function is designed to handle various SQuAD-like JSON formats, including nested structures and potential schema inconsistencies. 
    It ensures that the output dataset is always in the expected format, even if the input data has missing fields or unexpected nesting. 
    Args : data_path (str): The file path to the input JSON dataset. The function expects a SQuAD-like structure but can handle variations.
    Returns : Generator yielding dictionaries with keys 'id', 'context', 'question', and 'answers' (which contains 'text' and 'answer_start').
    """
    with open(data_path, 'r') as f:
        raw_data = json.load(f)
        data_list = raw_data['data'] if 'data' in raw_data else raw_data
        for entry in data_list:
            for para in entry.get("paragraphs", []):
                context = str(para.get("context", ""))
                for qa in para.get("qas", []):
                    raw_ans = qa.get("answers", [])
                    clean_texts, clean_starts = [], []
                    # SQuAD 2.0 Logic: Extract only if not impossible
                    if not qa.get("is_impossible", False) and isinstance(raw_ans, list):
                        for a in raw_ans:
                            t = a.get("text", "")
                            t = str(t[0]) if isinstance(t, list) and t else str(t)
                            s = a.get("answer_start", 0)
                            s = int(s[0]) if isinstance(s, list) and s else int(s)
                            if t and t in context:
                                clean_texts.append(t)
                                clean_starts.append(s)
                    yield {
                        "id": str(qa.get("id", "")),
                        "context": context,
                        "question": str(qa.get("question", "")),
                        "answers": {"text": clean_texts, "answer_start": clean_starts}
                    }


def preprocess_training_examples(examples, tokenizer):
    """
    This function preprocesses training examples for question answering tasks. 
    It tokenizes the input questions and contexts, and computes the start and end positions of the answers within the tokenized input. 
    The function is designed to handle cases where the answer may not be present in the context or when the input exceeds the model's maximum token limit, using a sliding window approach.
    Args : examples (dict): A batch of examples containing 'question', 'context', and 'answers' fields. tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be used for tokenizing the inputs.
    Returns : A dictionary containing tokenized inputs along with 'start_positions' and 'end_positions' for each example, which indicate the position of the answer in the tokenized input.
    """

    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions, examples["context"],
        max_length=384, truncation="only_second",
        return_overflowing_tokens=True, return_offsets_mapping=True,
        stride=128, padding="max_length",
    )

    sample_mapping = inputs.pop("overflow_to_sample_mapping")
    offset_mapping = inputs.pop("offset_mapping")
    start_positions, end_positions = [], []

    for i, sample_idx in enumerate(sample_mapping):
        offsets = offset_mapping[i]
        seq_ids = inputs.sequence_ids(i)

        # Locate Context
        c_start = 0
        while c_start < len(seq_ids) and seq_ids[c_start] != 1: c_start += 1
        c_end = c_start
        while c_end < len(seq_ids) and seq_ids[c_end] == 1: c_end += 1
        c_end -= 1

        ans = examples["answers"][sample_idx]
        if c_start >= len(seq_ids) or not ans["text"]:
            start_positions.append(0)
            end_positions.append(0)
        else:
            start_char = ans["answer_start"][0]
            end_char = start_char + len(ans["text"][0])

            if offsets[c_start][0] > start_char or offsets[c_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                s_idx = c_start
                while s_idx <= c_end and offsets[s_idx][0] <= start_char: s_idx += 1
                start_positions.append(s_idx - 1)
                e_idx = c_end
                while e_idx >= c_start and offsets[e_idx][1] >= end_char: e_idx -= 1
                end_positions.append(e_idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs



def basic_sci_normalize(s):
    """
    This function performs basic normalization of scientific text for evaluation purposes. 
    It normalizes unicode characters, converts text to lowercase, removes articles, and strips punctuation (except for a few that are relevant in scientific contexts). 
    Args : s (str): The input string to be normalized.
    Returns : A normalized version of the input string, suitable for comparison in evaluation metrics like F
    
    """


    if not s: return ""
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    exclude = set(string.punctuation) - {'.', ',', '-', '+', '(', ')', '/'}
    return ' '.join(''.join(ch for ch in s if ch not in exclude).split())

def run_benchmark(name, model_path, tokenizer, test_dataset_raw):
    """
    This function runs a comprehensive benchmark for a given question-answering model on a test dataset. 
    It evaluates the model's performance using multiple metrics, including SQuAD F1 and Exact Match, BLEU, ROUGE-L, and BERTScore. The function handles cases where the ground truth answers may be empty and ensures that the evaluation is robust against such scenarios.
    Args : 
        name (str): The name of the model being benchmarked. 
        model_path (str): The file path to the fine-tuned model. 
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer associated with the model.
        test_dataset_raw (list): A list of raw test examples, where each example is a dictionary containing 'id', 'context', 'question', and 'answers'.
    
    Returns : A dictionary containing the evaluation metrics for the model, including F1 score, Exact Match, BLEU, ROUGE-L, and BERTScore.
    """



    print(f"Benchmarking {name}...")
    squad_m = evaluate.load("squad"); bleu_m = evaluate.load("bleu")
    rouge_m = evaluate.load("rouge"); bert_m = evaluate.load("bertscore")

    qa_pipe = pipeline("question-answering", model=model_path, tokenizer=tokenizer,
                       device=0 if torch.cuda.is_available() else -1)

    preds_norm, refs_norm = [], []
    preds_raw, refs_raw = [], []

    for example in test_dataset_raw:
        res = qa_pipe(question=example["question"], context=example["context"])
        p_text = res["answer"]

        # FIX: Ensure reference is never an empty list to avoid ValueError in max()
        r_list = example["answers"]["text"] if example["answers"]["text"] else [""]

        preds_norm.append({"id": example["id"], "prediction_text": basic_sci_normalize(p_text)})
        refs_norm.append({"id": example["id"], "answers": {
            "text": [basic_sci_normalize(t) for t in r_list],
            "answer_start": example["answers"]["answer_start"] if example["answers"]["answer_start"] else [0]
        }})

        preds_raw.append(p_text)
        refs_raw.append(r_list[0])

    squad_res = squad_m.compute(predictions=preds_norm, references=refs_norm)

    # Filter empty ground truths for semantic metrics
    eval_pairs = [(p, r) for p, r in zip(preds_raw, refs_raw) if r.strip() != ""]
    if not eval_pairs:
        return {"model": name, "f1": squad_res["f1"], "em": squad_res["exact_match"], "bleu": 0, "rougeL": 0, "bertscore": 0}

    p_eval = [x[0] for x in eval_pairs]; r_eval = [x[1] for x in eval_pairs]
    bleu = bleu_m.compute(predictions=p_eval, references=[[r] for r in r_eval])
    rouge = rouge_m.compute(predictions=p_eval, references=r_eval)
    bert = bert_m.compute(predictions=p_eval, references=r_eval, lang="en")

    stats = {
        "model": name, "f1": squad_res["f1"], "em": squad_res["exact_match"],
        "bleu": bleu["bleu"], "rougeL": rouge["rougeL"], "bertscore": np.mean(bert["f1"])
    }
    print(f"{name} Metrics: F1={stats['f1']:.2f}, BERTScore={stats['bertscore']:.4f}")
    return stats



def main():
    data_path = "/content/drive/MyDrive/dsit/output_fg/CDE-QA_v2.json" # Update this path to your dataset
    full_ds = Dataset.from_generator(robust_gen, gen_kwargs={"data_path": data_path}, features=FEATURES)
    splits = full_ds.train_test_split(test_size=0.1, seed=42)

    models_config = {
    "VanillaBERT-Uncased": {"checkpoint": "bert-base-uncased", "lr": 2e-5, "epochs": 3},
    "VanillaBERT-Cased":   {"checkpoint": "bert-base-cased",   "lr": 2e-5, "epochs": 3},
    "BatteryBERT-Uncased": {"checkpoint": "batterydata/batterybert-uncased-squad-v1", "lr": 5e-6, "epochs": 3},
    "BatteryBERT-Cased":   {"checkpoint": "batterydata/batterybert-cased-squad-v1",   "lr": 5e-6, "epochs": 3},
}

    all_benchmarks = []
    for name, cfg in models_config.items():
        model_save_path = f"./model_{name}"
        tokenizer = AutoTokenizer.from_pretrained(cfg["checkpoint"])

        # RESUME CHECK: Skip training if the model already exists
        if os.path.exists(os.path.join(model_save_path, "config.json")):
            print(f"Found saved {name}, skipping to benchmark...")
            model = AutoModelForQuestionAnswering.from_pretrained(model_save_path)
        else:
            print(f"Training {name}...")
            model = AutoModelForQuestionAnswering.from_pretrained(cfg["checkpoint"])
            train_ds = splits["train"].map(lambda x: preprocess_training_examples(x, tokenizer),
                                           batched=True, remove_columns=splits["train"].column_names)

            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=f"./checkpoints_{name}", num_train_epochs=cfg["epochs"],
                    learning_rate=cfg["lr"], per_device_train_batch_size=12,
                    fp16=True, save_strategy="no", report_to="none"
                ),
                train_dataset=train_ds,
                data_collator=DefaultDataCollator(),
            )
            trainer.train()
            trainer.save_model(model_save_path)
            tokenizer.save_pretrained(model_save_path)

        # Benchmark on test split
        all_benchmarks.append(run_benchmark(name, model_save_path, tokenizer, splits["test"]))

    print("FINAL COMPARISON")
    print(pd.DataFrame(all_benchmarks).to_string())

if __name__ == "__main__":
    main()