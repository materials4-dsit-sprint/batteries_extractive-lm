import os
import time
import torch
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering, 
    TrainingArguments, 
    Trainer, 
    DefaultDataCollator, 
    pipeline
)
import evaluate

""" Loading Evaluation metrics and Datasets for Benchmarking """


print("Loading datasets and metrics...")
data_path = "/content/drive/MyDrive/dsit/output_squad/squad_dataset.json"
raw_datasets = load_dataset("json", data_files=data_path, field="data")

squad_metric = evaluate.load("squad")
bleu_metric = evaluate.load("bleu")
bertscore_metric = evaluate.load("bertscore")


def flatten_squad(batch):

    """
    This function takes a batch of the original SQuAD dataset format and flattens it into a more straightforward structure.
    It iterates through the nested structure of the dataset, extracting the context, question, and
    
    Args : batch (dict): A batch of the original SQuAD dataset format, which contains a list of papers, each with paragraphs and Q&A pairs.
    Returns : dict: A flattened version of the dataset with separate lists for 'id', 'context', 'question', and 'answers'.

    """
    new_rows = []
    for paper in batch["paragraphs"]:
        for para in paper:
            context = para["context"]
            for qa in para["qas"]:
                new_rows.append({
                    "id": str(qa.get("id", len(new_rows))),
                    "context": context,
                    "question": qa["question"],
                    "answers": qa["answers"]
                })
    return {"id": [r["id"] for r in new_rows],
            "context": [r["context"] for r in new_rows],
            "question": [r["question"] for r in new_rows],
            "answers": [r["answers"] for r in new_rows]}

def preprocess_training_examples(examples, tokenizer):
    """
    This function preprocesses the training examples for question answering. It tokenizes the questions and contexts,
    and computes the start and end positions of the answers in the tokenized input. It handles cases where the answer may not be fully contained within the tokenized context, assigning a default position in
    
    Args : examples (dict): A batch of examples containing 'question', 'context', and 'answers'. 
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenizing the input.

    Returns : dict: A dictionary containing the tokenized inputs along with 'start_positions' and 'end_positions' for the answers.

    """
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=512, # Increased to max for BERT to capture more context
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )
    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions, end_positions = [], []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i][0]
        start_char = answer["answer_start"]
        end_char = start_char + len(answer["text"])
        sequence_ids = inputs.sequence_ids(i)

        idx = 0
        while sequence_ids[idx] != 1: idx += 1
        context_start = idx
        while sequence_ids[idx] == 1: idx += 1
        context_end = idx - 1

        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char: idx += 1
            start_positions.append(idx - 1)
            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char: idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

def run_benchmark(name, model_path, tokenizer, test_dataset_raw):

    """ This function benchmarks a given question-answering model on a test dataset. 
    It uses the Hugging Face pipeline for question answering to generate predictions, 
    and then computes various evaluation metrics such as F1 score, Exact Match, BLEU, and BERTScore. 
    It also measures the latency of the model's predictions.
    
    Args : name (str): The name of the model being benchmarked.
            model_path (str): The path to the trained model to be evaluated.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer associated with the model.
            test_dataset_raw (datasets.Dataset): The raw test dataset containing 'question', 'context', and 'answers'.
    
    Returns : dict: A dictionary containing the evaluation results, including F1 score, Exact Match, BLEU, BERTScore, and latency in milliseconds.
    
    """


    print(f"Benchmarking {name}...")
    qa_pipeline = pipeline("question-answering", model=model_path, tokenizer=tokenizer, 
                           device=0 if torch.cuda.is_available() else -1)
    
    predictions, references, latencies = [], [], []

    for example in test_dataset_raw:
        start_time = time.time()
        res = qa_pipeline(question=example["question"], context=example["context"])
        latencies.append(time.time() - start_time)
        
        predictions.append({"id": example["id"], "prediction_text": res["answer"]})
        references.append({
            "id": example["id"], 
            "answers": {
                "text": [ans["text"] for ans in example["answers"]],
                "answer_start": [ans["answer_start"] for ans in example["answers"]]
            }
        })

    results = squad_metric.compute(predictions=predictions, references=references)
    pred_strings = [p["prediction_text"] for p in predictions]
    ref_strings = [r["answers"]["text"][0] for r in references]
    
    bleu = bleu_metric.compute(predictions=pred_strings, references=ref_strings)
    b_score = bertscore_metric.compute(predictions=pred_strings, references=ref_strings, lang="en")
    
    return {
        "model": name,
        "f1": results["f1"],
        "em": results["exact_match"],
        "bleu": bleu["bleu"],
        "bertscore": np.mean(b_score["f1"]),
        "latency_ms": np.mean(latencies) * 1000
    }


def plot_results(all_trainers, benchmark_stats):

    """
    Plotting function to visualize training loss and benchmark results. 
    It creates a line plot for training loss across epochs for each model, 
    and a radar chart to compare the evaluation metrics (F1, Exact Match, BLEU, BERTScore) 
    across different models.

    Args : all_trainers (dict): A dictionary containing the Trainer objects for each model, used to extract training logs.
            benchmark_stats (list): A list of dictionaries containing the evaluation results for each model, used
                                    to create the radar chart.

    Returns : None: This function generates plots and does not return any value.
    
    """
    # Loss Plot
    plt.figure(figsize=(10, 5))
    for name, trainer in all_trainers.items():
        logs = [x for x in trainer.state.log_history if 'loss' in x]
        epochs = [x['epoch'] for x in logs]
        losses = [x['loss'] for x in logs]
        plt.plot(epochs, losses, label=name)
    plt.title("Epochs vs Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Radar Plot
    labels = ['F1', 'EM', 'BERTScore', 'BLEU']
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for stat in benchmark_stats:
        # Normalize scores to 0-1 for radar chart
        values = [stat['f1']/100, stat['em']/100, stat['bertscore'], stat['bleu']]
        values += values[:1]
        ax.plot(angles, values, label=stat['model'])
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.show()


def main():
    print("Flattening dataset...")
    full_ds = raw_datasets["train"].map(flatten_squad, batched=True, remove_columns=raw_datasets["train"].column_names)

    #full_ds = load_dataset("rajpurkar/squad")
    #splits = full_ds["train"].train_test_split(train_size=800, test_size=200, seed=42)
    # --- 80:10:10 SPLIT ---
    train_test_split = full_ds.train_test_split(test_size=0.2, seed=42)
    test_val_split = train_test_split['test'].train_test_split(test_size=0.5, seed=42)
    
    train_ds_raw = train_test_split['train']
    val_ds_raw = test_val_split['train']
    test_ds_raw = test_val_split['test']

    models_to_check = {
        "VanillaBERT": "bert-base-uncased",
        "SciBERT": "allenai/scibert_scivocab_uncased",
        "MatSciBERT": "m3rg-iitd/matscibert",
        "BatteryBERT": "batterydata/batterybert-uncased-squad-v1"
    }

    all_benchmarks = []
    all_trainers = {}

    for name, checkpoint in models_to_check.items():
        print(f"Training: {name}")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)
        
        train_dataset = train_ds_raw.map(preprocess_training_examples, batched=True, fn_kwargs={"tokenizer": tokenizer})
        val_dataset = val_ds_raw.map(preprocess_training_examples, batched=True, fn_kwargs={"tokenizer": tokenizer})

        training_args = TrainingArguments(
            output_dir=f"./results_{name}",
            eval_strategy="epoch", # Changed to see validation loss per epoch
            learning_rate=2e-5,           # Lowered for stability
            per_device_train_batch_size=8,
            num_train_epochs=5,           # Increased for better convergence
            weight_decay=0.01,
            warmup_ratio=0.1,             # Linear warmup to prevent early divergence
            fp16=True if torch.cuda.is_available() else False,
            logging_steps=10,
            save_strategy="no"
        )

        trainer = Trainer(
            model=model, args=training_args,
            train_dataset=train_dataset, eval_dataset=val_dataset,
            data_collator=DefaultDataCollator(),
        )

        trainer.train()
        all_trainers[name] = trainer
        
        # Benchmarking on the 10% TEST SET
        model_path = f"./model_{name}"
        trainer.save_model(model_path)
        tokenizer.save_pretrained(model_path)
        
        stats = run_benchmark(name, model_path, tokenizer, test_ds_raw)
        all_benchmarks.append(stats)

    # Visualization
    plot_results(all_trainers, all_benchmarks)

if __name__ == "__main__":
    main()









