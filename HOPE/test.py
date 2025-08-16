from transformers import BartForConditionalGeneration, BartTokenizer
from datasets import load_from_disk
import evaluate
from rouge import Rouge
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import numpy as np
import argparse


def load_model_and_tokenizer(model_path):
    """Load the fine-tuned model and tokenizer"""
    model = BartForConditionalGeneration.from_pretrained(model_path)
    tokenizer = BartTokenizer.from_pretrained(model_path)
    return model, tokenizer


def generate_batch_responses(model, tokenizer, batch_texts, max_length=128):
    """Generate responses for a batch of input texts"""
    inputs = tokenizer(
        batch_texts, max_length=512, truncation=True, padding=True, return_tensors="pt"
    )

    if torch.cuda.is_available():
        model = model.cuda()
        inputs = {k: v.cuda() for k, v in inputs.items()}

    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )

    decoded_outputs = [
        tokenizer.decode(output, skip_special_tokens=True) for output in outputs
    ]

    return decoded_outputs


def create_dataloader(dataset, batch_size):
    """Create a dataloader for the dataset"""

    def collate_fn(batch):
        input_texts = [item["input_text"] for item in batch]
        target_texts = [item["target_text"] for item in batch]
        return input_texts, target_texts

    return DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False
    )


def calculate_sentence_bleu_batch(predictions, references):
    """Calculate sentence BLEU scores for batches of predictions and references"""
    bleu_scores = []
    smoother = SmoothingFunction()

    for pred, ref in zip(predictions, references):
        # Tokenize prediction and reference
        pred_tokens = word_tokenize(pred.lower())
        ref_tokens = [word_tokenize(ref.lower())]  # BLEU expects list of references

        # Calculate BLEU score with smoothing
        try:
            bleu = sentence_bleu(
                ref_tokens,
                pred_tokens,
                weights=(0.5, 0.5),
                smoothing_function=smoother.method1,
            )
        except Exception:
            bleu = 0.0

        bleu_scores.append(bleu)

    return bleu_scores


def get_rouge_l(preds, refs):
    """Calculate average ROUGE-L F1 score for a batch of predictions and references"""
    rouge = Rouge()
    scores = [
        rouge.get_scores(pred, ref, avg=True)["rouge-l"]["f"]
        for pred, ref in zip(preds, refs)
    ]
    return np.mean(scores)


def evaluate_model(model_path, test_data_path, batch_size=16, num_samples=None):
    """Evaluate model using BERTScore and BLEU with batch processing"""
    # Load model and test data
    model, tokenizer = load_model_and_tokenizer(model_path)
    test_dataset = load_from_disk(test_data_path)

    # Load BERTScore metric
    metric = evaluate.load("bertscore", trust_remote_code=True)

    if num_samples:
        test_dataset = test_dataset.select(range(num_samples))

    dataloader = create_dataloader(test_dataset, batch_size)

    predictions = []
    references = []

    model.eval()
    with torch.no_grad():
        for input_texts, target_texts in tqdm(dataloader, desc="Generating responses"):
            batch_predictions = generate_batch_responses(model, tokenizer, input_texts)

            predictions.extend(batch_predictions)
            references.extend(target_texts)

            # Calculate BERTScore for current batch
            # batch_results = metric.compute(
            #     predictions=batch_predictions, references=target_texts, lang="en"
            # )

            # Optionally print batch results
            # batch_f1 = np.mean(batch_results["f1"])
            # print(f"Batch BERTScore F1: {batch_f1:.4f}")

    # Calculate final BERTScore
    print("\nCalculating final BERTScore...")
    bert_results = metric.compute(
        predictions=predictions, references=references, lang="en"
    )
    print("\nCalculating final RougeScore...")
    mean_rouge = get_rouge_l(predictions, references)
    # Calculate BLEU scores

    print("\nCalculating BLEU scores...")
    bleu_scores = calculate_sentence_bleu_batch(predictions, references)
    mean_bleu = np.mean(bleu_scores)

    # Calculate mean BERTScores
    mean_precision = np.mean(bert_results["precision"])
    mean_recall = np.mean(bert_results["recall"])
    mean_f1 = np.mean(bert_results["f1"])

    # Print results
    print("\nEvaluation Results:")
    print("\nBERTScore Results:")
    print(f"Precision: {mean_precision:.4f}")
    print(f"Recall: {mean_recall:.4f}")
    print(f"F1: {mean_f1:.4f}")

    print("\nBLEU Score Results:")
    print(f"Mean BLEU: {mean_bleu:.4f}")

    print("\nRouge Score Results:")
    print(f"Mean Rouge: {mean_rouge}")

    return {
        "bertscore": {
            "precision": bert_results["precision"],
            "recall": bert_results["recall"],
            "f1": bert_results["f1"],
            "mean_precision": mean_precision,
            "mean_recall": mean_recall,
            "mean_f1": mean_f1,
        },
        "bleu": {"mean_bleu": mean_bleu, "individual_scores": bleu_scores},
        "predictions": predictions,
        "references": references,
    }


if __name__ == "__main__":
    # Configuration
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="model path",
    )
    args = parser.parse_args()
    MODEL_PATH = args.model
    TEST_DATA_PATH = ""
    BATCH_SIZE = 16  # Adjust based on your GPU memory
    NUM_SAMPLES = None

    # Run evaluation
    results = evaluate_model(
        MODEL_PATH, TEST_DATA_PATH, batch_size=BATCH_SIZE, num_samples=NUM_SAMPLES
    )
