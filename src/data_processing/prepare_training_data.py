import json
import argparse
from pathlib import Path
import random

# Define the prompt template directly
PROMPT_TEMPLATE = '''This is an excerpt from a novel. Write the next excerpt of similar length. Use the same style as the excerpt. Make sure that while stylistically similar, the new section moves the story forward and/or develops the characters and/or adds new information or in some way continues on meaningfully from the previous section.

EXCERPT:
{}'''

def save_to_jsonl(data: list, file_path: Path):
    """Saves a list of strings to a JSONL file, with each string under the 'text' key."""
    with file_path.open("w", encoding="utf-8") as f:
        for item in data:
            # Ensure the output is a valid JSON line
            f.write(json.dumps({"text": item}) + '\n')

if __name__ == "__main__":
    DEFAULT_INPUT_FILE = "semantic_chunks.json"
    DEFAULT_OUTPUT_DIR = "."
    DEFAULT_TRAIN_RATIO = 0.85
    DEFAULT_SEED = 42

    parser = argparse.ArgumentParser(
        description="Prepare training and validation data from semantic chunks."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=DEFAULT_INPUT_FILE,
        help=f"Path to the input JSON file containing semantic chunks (default: {DEFAULT_INPUT_FILE})."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save the train.jsonl and valid.jsonl files (default: {DEFAULT_OUTPUT_DIR})."
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help=f"Proportion of data to use for the training set (default: {DEFAULT_TRAIN_RATIO})."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for shuffling data (default: {DEFAULT_SEED})."
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure output directory exists

    if not input_path.is_file():
        print(f"Error: Input file not found at {input_path}")
        exit(1)

    print(f"Loading chunks from: {input_path}")
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "chunks" not in data or not isinstance(data["chunks"], list):
        print(f"Error: Input file {input_path} does not contain a list under the 'chunks' key.")
        exit(1)

    chunks = data["chunks"]
    print(f"Loaded {len(chunks)} chunks.")

    if len(chunks) < 2:
        print("Error: Need at least 2 chunks to create prompt/completion pairs.")
        exit(1)

    print("Creating prompt/completion pairs...")
    paired_texts = []
    # Iterate up to the second-to-last chunk to form pairs
    for i in range(len(chunks) - 1):
        prompt_part = PROMPT_TEMPLATE.format(chunks[i])
        completion_part = chunks[i+1]
        combined_text = f"{prompt_part}\nNEXT EXCERPT:\n{completion_part}"
        paired_texts.append(combined_text)

    print(f"Created {len(paired_texts)} pairs.")

    # Shuffle the paired texts
    print(f"Shuffling {len(paired_texts)} pairs with seed {args.seed}...")
    random.seed(args.seed)
    random.shuffle(paired_texts)

    # Split the shuffled pairs
    split_index = int(len(paired_texts) * args.train_ratio)
    train_data = paired_texts[:split_index]
    valid_data = paired_texts[split_index:]

    print(f"Splitting into {len(train_data)} training samples and {len(valid_data)} validation samples.")

    train_output_path = output_dir / "train.jsonl"
    valid_output_path = output_dir / "valid.jsonl"

    print(f"Saving training data to: {train_output_path}")
    save_to_jsonl(train_data, train_output_path)

    print(f"Saving validation data to: {valid_output_path}")
    save_to_jsonl(valid_data, valid_output_path)

    print("Done.")

"""
Example Usage:
python src/data_processing/prepare_training_data.py \
    --input_file semantic_chunks.json \
    --output_dir DATA/processed_novel \
    --train_ratio 0.9 \
    --seed 123
""" 