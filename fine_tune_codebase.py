import os
import argparse
import fnmatch
import logging
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    pipeline,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
)
from datasets import load_dataset, load_from_disk
import evaluate
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="fine_tune.log",
)
logger = logging.getLogger(__name__)

# Read .codeignore patterns
def read_ignore_patterns(ignore_file):
    """
    Reads patterns from a .codeignore file and returns a list of patterns.
    """
    if not os.path.exists(ignore_file):
        return []
    with open(ignore_file, "r") as f:
        patterns = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return patterns

# Check if a file or directory should be ignored
def should_ignore(path, patterns):
    """
    Checks if a file or directory matches any of the ignore patterns.
    """
    for pattern in patterns:
        if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(os.path.basename(path), pattern):
            return True
    return False

# Preprocess dataset (e.g., remove comments)
def remove_comments(code):
    """
    Removes single-line and multi-line comments from code.
    """
    import re
    code = re.sub(r"//.*?\n", "\n", code)  # Single-line comments
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)  # Multi-line comments
    return code

# Prepare the dataset
def prepare_dataset(input_dir, output_file, ignore_file=".codeignore", file_extensions=None, preprocess=False):
    """
    Concatenates all files in the codebase directory into a single text file,
    excluding files and folders that match patterns in .codeignore.
    """
    ignore_patterns = read_ignore_patterns(ignore_file)
    all_files = []

    # Collect all files first
    for root, dirs, files in os.walk(input_dir):
        dirs[:] = [d for d in dirs if not should_ignore(os.path.join(root, d), ignore_patterns)]
        for file in files:
            file_path = os.path.join(root, file)
            if should_ignore(file_path, ignore_patterns):
                continue
            if file_extensions and not any(file.endswith(ext) for ext in file_extensions):
                continue
            all_files.append(file_path)

    # Process files with a progress bar
    with open(output_file, "w") as outfile:
        for file_path in tqdm(all_files, desc="Processing files"):
            try:
                with open(file_path, "r") as infile:
                    content = infile.read()
                    if preprocess:
                        content = remove_comments(content)
                    outfile.write(f"<file_start>{file_path}\n")
                    outfile.write(content)
                    outfile.write("\n<file_end>\n\n")
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
    logger.info(f"Dataset prepared and saved to {output_file}")

# Load and tokenize the dataset
def load_and_tokenize_dataset(output_file, model_name):
    """
    Loads the dataset and tokenizes it using the model's tokenizer.
    """
    if os.path.exists("tokenized_dataset"):
        logger.info("Loading tokenized dataset from disk...")
        tokenized_dataset = load_from_disk("tokenized_dataset")
    else:
        logger.info("Tokenizing dataset...")
        dataset = load_dataset("text", data_files={"train": output_file})
        dataset = dataset["train"].train_test_split(test_size=0.1)  # 90% train, 10% validation

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            logger.warning(
                f"Tokenizer for '{model_name}' does not have a pad_token. "
                f"Setting pad_token to eos_token ('{tokenizer.eos_token}') for padding."
            )
            tokenizer.pad_token = tokenizer.eos_token

        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        tokenized_dataset.save_to_disk("tokenized_dataset")
    return tokenized_dataset, tokenizer

# Fine-tune the model with LoRA
def fine_tune_model(tokenized_dataset, tokenizer, model_name, output_dir, learning_rate, batch_size, num_epochs, early_stopping_patience, gradient_accumulation_steps, fp16, resume_from_checkpoint, tensorboard, quantize):
    """
    Fine-tunes the model using the tokenized dataset with LoRA.
    """
    logger.info("Loading base model...")

    # Quantization (optional)
    quantization_config = None
    if quantize:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,  # Quantize to 8-bit
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            logger.warning("Tokenizer passed to fine_tune_model has no pad_token. Setting to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError("Tokenizer has no pad_token and no eos_token to use as a fallback.")

    if model.config.pad_token_id is None or model.config.pad_token_id != tokenizer.pad_token_id:
        logger.info(
            f"Aligning model's pad_token_id ({model.config.pad_token_id}) "
            f"with tokenizer's pad_token_id ({tokenizer.pad_token_id})."
        )
        model.config.pad_token_id = tokenizer.pad_token_id

    # Apply LoRA
    lora_config = LoraConfig(
        r=8,  # Rank of the low-rank matrices
        lora_alpha=32,  # Scaling factor
        target_modules=["q_proj", "v_proj"],  # Target layers
        lora_dropout=0.1,
    )
    model = get_peft_model(model, lora_config)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=500,
        evaluation_strategy="epoch",
        fp16=fp16,  # Mixed precision
        save_strategy="epoch",  # Save checkpoints
        resume_from_checkpoint=resume_from_checkpoint,  # Resume training
        report_to="tensorboard" if tensorboard else None,  # TensorBoard logging
    )

    # Load evaluation metrics
    metric = evaluate.load("perplexity")
    bleu = evaluate.load("bleu")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        perplexity = metric.compute(predictions=predictions, references=labels)
        bleu_score = bleu.compute(predictions=predictions, references=labels)
        return {"perplexity": perplexity["perplexity"], "bleu": bleu_score["bleu"]}

    # Initialize the Trainer
    logger.info("Starting fine-tuning...")
    callbacks = []
    if early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # Start training
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Fine-tuned model saved to {output_dir}")

# Test the fine-tuned model
def test_model(output_dir, tokenizer, prompt):
    """
    Tests the fine-tuned model by generating code from a prompt.
    """
    logger.info("Testing fine-tuned model...")
    generator = pipeline("text-generation", model=output_dir, tokenizer=tokenizer)
    output = generator(prompt, max_length=100)
    print("Generated Code:\n", output[0]["generated_text"])

# Interactive testing
def interactive_test(output_dir, tokenizer):
    """
    Allows interactive testing of the fine-tuned model.
    """
    logger.info("Starting interactive testing mode...")
    generator = pipeline("text-generation", model=output_dir, tokenizer=tokenizer)
    print("Interactive testing mode. Enter a prompt or type 'exit' to quit.")
    while True:
        prompt = input(">>> ")
        if prompt.lower() == "exit":
            break
        output = generator(prompt, max_length=100)
        print("Generated Code:\n", output[0]["generated_text"])

# Main function
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Fine-tune a language model on a codebase.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the codebase files.")
    parser.add_argument("--output_file", type=str, default="codebase.txt", help="Output file for the concatenated dataset.")
    parser.add_argument("--model_name", type=str, default="codellama/CodeLlama-7b-hf", help="Base model to fine-tune.")
    parser.add_argument("--output_dir", type=str, default="./fine_tuned_model", help="Directory to save the fine-tuned model.")
    parser.add_argument("--test_prompt", type=str, default="// Create a new function", help="Prompt to test the fine-tuned model.")
    parser.add_argument("--ignore_file", type=str, default=".codeignore", help="File containing ignore patterns.")
    parser.add_argument("--file_extensions", type=str, nargs="+", default=None, help="File extensions to include (e.g., .py .js).")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for training.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Number of epochs to wait before early stopping.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients.")
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision training.")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Resume training from the latest checkpoint.")
    parser.add_argument("--tensorboard", action="store_true", help="Log metrics to TensorBoard.")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess the dataset (e.g., remove comments).")
    parser.add_argument("--quantize", action="store_true", help="Quantize the model to 8-bit.")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive testing mode.")
    args = parser.parse_args()

    # Prepare the dataset
    prepare_dataset(args.input_dir, args.output_file, args.ignore_file, args.file_extensions, args.preprocess)

    # Load and tokenize the dataset
    tokenized_dataset, tokenizer = load_and_tokenize_dataset(args.output_file, args.model_name)

    # Fine-tune the model
    fine_tune_model(
        tokenized_dataset,
        tokenizer,
        args.model_name,
        args.output_dir,
        args.learning_rate,
        args.batch_size,
        args.num_epochs,
        args.early_stopping_patience,
        args.gradient_accumulation_steps,
        args.fp16,
        args.resume_from_checkpoint,
        args.tensorboard,
        args.quantize,
    )

    # Test the fine-tuned model
    if args.interactive:
        interactive_test(args.output_dir, tokenizer)
    else:
        test_model(args.output_dir, tokenizer, args.test_prompt)

if __name__ == "__main__":
    main()
