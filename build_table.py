#!/usr/bin/env python3
"""
Build precomputed trigram probability tables from a text corpus.

This is the offline phase: run once to produce a static lookup table that
the compressor uses for zero-inference compression.

Usage:
    python build_table.py --corpus wikitext-103 --output trigram_table.npz
    python build_table.py --corpus wikitext-2 --output trigram_table.npz
    python build_table.py --file my_corpus.txt --output trigram_table.npz
"""

import argparse
import os
import sys
import time

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


# --- Configuration ---
DEFAULT_TOKENIZER = "HuggingFaceTB/SmolLM2-135M"

# Top-K entries to store per context
BIGRAM_TOP_K = 256
TRIGRAM_TOP_K = 128

# Maximum number of contexts to keep (prune rare ones)
MAX_BIGRAM_CONTEXTS = 100_000
MAX_TRIGRAM_CONTEXTS = 2_000_000

# Smoothing: add-delta for unseen n-grams
UNIGRAM_DELTA = 0.01
BIGRAM_DELTA = 0.001
TRIGRAM_DELTA = 0.0001


def load_corpus_wikitext(variant: str = "wikitext-103-raw-v1") -> list[str]:
    """Load WikiText corpus from Hugging Face datasets."""
    from datasets import load_dataset

    print(f"Loading {variant} from Hugging Face...", file=sys.stderr)
    ds = load_dataset("Salesforce/wikitext", variant, split="train")
    texts = [row["text"] for row in ds if row["text"].strip()]
    print(f"Loaded {len(texts)} non-empty lines", file=sys.stderr)
    return texts


def load_corpus_file(path: str) -> list[str]:
    """Load corpus from a local text file."""
    print(f"Loading corpus from {path}...", file=sys.stderr)
    with open(path, "r", encoding="utf-8") as f:
        texts = [line for line in f if line.strip()]
    print(f"Loaded {len(texts)} non-empty lines", file=sys.stderr)
    return texts


def load_corpus_huggingface(dataset_name: str, languages: list[str] = None,
                             max_samples: int = None) -> list[str]:
    """Load corpus from HuggingFace dataset with optional language filtering.

    Args:
        dataset_name: HuggingFace dataset identifier (e.g., "codeparrot/github-code")
        languages: List of languages to filter (e.g., ["HTML", "CSS", "JavaScript", "JSON"])
        max_samples: Maximum number of samples to load (None = all)

    Returns:
        List of text strings
    """
    from datasets import load_dataset

    print(f"Loading dataset: {dataset_name}", file=sys.stderr)
    if languages:
        print(f"  Filtering languages: {', '.join(languages)}", file=sys.stderr)

    # Load dataset
    ds = load_dataset(dataset_name, split="train", streaming=True)

    texts = []
    count = 0

    for row in tqdm(ds, desc="Loading samples", file=sys.stderr):
        # Check language filter if specified
        if languages:
            lang = row.get("language", "").lower()
            if not any(l.lower() == lang for l in languages):
                continue

        # Extract code/text content (field name varies by dataset)
        content = row.get("code") or row.get("content") or row.get("text") or ""

        if content.strip():
            texts.append(content)
            count += 1

            if max_samples and count >= max_samples:
                break

    print(f"Loaded {len(texts)} samples", file=sys.stderr)
    return texts


def tokenize_corpus(texts: list[str], tokenizer) -> np.ndarray:
    """Tokenize all texts and return a flat array of token IDs."""
    print("Tokenizing corpus...", file=sys.stderr)

    all_ids = []
    total_tokens = 0

    for i in tqdm(range(0, len(texts), 1000), desc="Tokenizing", file=sys.stderr):
        batch = texts[i:i + 1000]
        # Join batch into a single string with newlines (preserves document structure)
        chunk = "\n".join(batch)
        ids = tokenizer.encode(chunk)
        all_ids.extend(ids)
        total_tokens += len(ids)

    tokens = np.array(all_ids, dtype=np.int32)
    print(f"Total tokens: {len(tokens):,}", file=sys.stderr)
    return tokens


def count_ngrams(tokens: np.ndarray, vocab_size: int):
    """Count unigram, bigram, and trigram frequencies.

    Uses packed integer keys for memory-efficient counting.
    Bigrams packed as: prev << 16 | curr (uint32)
    Trigrams packed as: prev2 << 32 | prev1 << 16 | curr (uint64)
    """
    n = len(tokens)
    print(f"Counting n-grams over {n:,} tokens (vocab={vocab_size})...",
          file=sys.stderr)

    # Unigram counts
    print("  Counting unigrams...", file=sys.stderr)
    unigram_counts = np.bincount(tokens, minlength=vocab_size).astype(np.int64)

    # Bigram counts using packed keys
    print("  Counting bigrams...", file=sys.stderr)
    prev_tokens = tokens[:-1].astype(np.uint32)
    curr_tokens = tokens[1:].astype(np.uint32)
    bigram_keys = (prev_tokens.astype(np.uint64) << 16) | curr_tokens.astype(np.uint64)
    bigram_keys_sorted = np.sort(bigram_keys)
    # Find unique keys and their counts
    bi_unique, bi_counts = np.unique(bigram_keys_sorted, return_counts=True)
    print(f"  Unique bigrams: {len(bi_unique):,}", file=sys.stderr)

    # Trigram counts using packed keys
    print("  Counting trigrams...", file=sys.stderr)
    prev2_tokens = tokens[:-2].astype(np.uint64)
    prev1_tokens = tokens[1:-1].astype(np.uint64)
    curr_tokens_tri = tokens[2:].astype(np.uint64)
    trigram_keys = (prev2_tokens << 32) | (prev1_tokens << 16) | curr_tokens_tri
    trigram_keys_sorted = np.sort(trigram_keys)
    tri_unique, tri_counts = np.unique(trigram_keys_sorted, return_counts=True)
    print(f"  Unique trigrams: {len(tri_unique):,}", file=sys.stderr)

    return unigram_counts, (bi_unique, bi_counts), (tri_unique, tri_counts)


def build_unigram_table(unigram_counts: np.ndarray, delta: float) -> np.ndarray:
    """Build unigram log-probability table with add-delta smoothing.

    Returns float32 array of shape (vocab_size,) with probabilities (NOT log).
    """
    smoothed = unigram_counts.astype(np.float64) + delta
    probs = smoothed / smoothed.sum()
    return probs.astype(np.float32)


def build_bigram_table(bi_unique, bi_counts, unigram_counts,
                       vocab_size, top_k, max_contexts):
    """Build bigram lookup table.

    For each context token, stores the top-K most likely next tokens
    and their conditional probabilities, plus the remaining mass.

    Returns:
        bigram_context_keys: uint16 array of context tokens (sorted)
        bigram_top_tokens: uint16 array of shape (num_contexts, top_k)
        bigram_top_probs: float32 array of shape (num_contexts, top_k)
        bigram_remaining_mass: float32 array of shape (num_contexts,)
    """
    print(f"Building bigram table (top-K={top_k}, max_ctx={max_contexts:,})...",
          file=sys.stderr)

    # Unpack bigram keys to get context and next token
    contexts = (bi_unique >> 16).astype(np.uint16)
    next_tokens = (bi_unique & 0xFFFF).astype(np.uint16)

    # Group by context: find unique contexts and their index ranges
    unique_contexts, ctx_starts = np.unique(contexts, return_index=True)
    ctx_ends = np.append(ctx_starts[1:], len(contexts))

    # Rank contexts by total count (keep most frequent)
    ctx_total_counts = np.array([
        bi_counts[ctx_starts[i]:ctx_ends[i]].sum()
        for i in range(len(unique_contexts))
    ])

    if len(unique_contexts) > max_contexts:
        top_ctx_indices = np.argsort(ctx_total_counts)[::-1][:max_contexts]
        top_ctx_indices = np.sort(top_ctx_indices)  # keep sorted order
    else:
        top_ctx_indices = np.arange(len(unique_contexts))

    num_ctx = len(top_ctx_indices)
    print(f"  Keeping {num_ctx:,} bigram contexts", file=sys.stderr)

    bigram_context_keys = np.empty(num_ctx, dtype=np.uint16)
    bigram_top_tokens = np.zeros((num_ctx, top_k), dtype=np.uint16)
    bigram_top_probs = np.zeros((num_ctx, top_k), dtype=np.float32)
    bigram_remaining_mass = np.zeros(num_ctx, dtype=np.float32)

    for out_i, ctx_i in enumerate(tqdm(top_ctx_indices, desc="  Bigram table",
                                       file=sys.stderr)):
        ctx_token = unique_contexts[ctx_i]
        start, end = ctx_starts[ctx_i], ctx_ends[ctx_i]

        these_tokens = next_tokens[start:end]
        these_counts = bi_counts[start:end].astype(np.float64)

        # Conditional probability with simple add-delta smoothing
        total = these_counts.sum() + BIGRAM_DELTA * vocab_size
        these_probs = these_counts / total

        # Get top-K
        k = min(top_k, len(these_tokens))
        if k < len(these_tokens):
            top_indices = np.argpartition(these_probs, -k)[-k:]
            top_indices = top_indices[np.argsort(these_probs[top_indices])[::-1]]
        else:
            top_indices = np.argsort(these_probs)[::-1]

        bigram_context_keys[out_i] = ctx_token
        bigram_top_tokens[out_i, :k] = these_tokens[top_indices]
        bigram_top_probs[out_i, :k] = these_probs[top_indices].astype(np.float32)
        bigram_remaining_mass[out_i] = 1.0 - these_probs[top_indices].sum()

    return bigram_context_keys, bigram_top_tokens, bigram_top_probs, bigram_remaining_mass


def build_trigram_table(tri_unique, tri_counts, vocab_size, top_k, max_contexts):
    """Build trigram lookup table.

    For each bigram context (prev2, prev1), stores the top-K most likely
    next tokens and their conditional probabilities, plus remaining mass.

    Returns:
        trigram_context_keys: uint32 array of packed (prev2 << 16 | prev1) (sorted)
        trigram_top_tokens: uint16 array of shape (num_contexts, top_k)
        trigram_top_probs: float32 array of shape (num_contexts, top_k)
        trigram_remaining_mass: float32 array of shape (num_contexts,)
    """
    print(f"Building trigram table (top-K={top_k}, max_ctx={max_contexts:,})...",
          file=sys.stderr)

    # Unpack trigram keys to get context pair and next token
    prev2 = (tri_unique >> 32).astype(np.uint16)
    prev1 = ((tri_unique >> 16) & 0xFFFF).astype(np.uint16)
    next_tokens = (tri_unique & 0xFFFF).astype(np.uint16)

    # Pack context pair as uint32 for grouping
    context_packed = (prev2.astype(np.uint32) << 16) | prev1.astype(np.uint32)

    # Group by context pair
    unique_ctx_packed, ctx_starts = np.unique(context_packed, return_index=True)
    ctx_ends = np.append(ctx_starts[1:], len(context_packed))

    # Rank contexts by total count
    ctx_total_counts = np.array([
        tri_counts[ctx_starts[i]:ctx_ends[i]].sum()
        for i in tqdm(range(len(unique_ctx_packed)), desc="  Ranking contexts",
                      file=sys.stderr)
    ])

    if len(unique_ctx_packed) > max_contexts:
        top_ctx_indices = np.argsort(ctx_total_counts)[::-1][:max_contexts]
        top_ctx_indices = np.sort(top_ctx_indices)
    else:
        top_ctx_indices = np.arange(len(unique_ctx_packed))

    num_ctx = len(top_ctx_indices)
    print(f"  Keeping {num_ctx:,} trigram contexts", file=sys.stderr)

    trigram_context_keys = np.empty(num_ctx, dtype=np.uint32)
    trigram_top_tokens = np.zeros((num_ctx, top_k), dtype=np.uint16)
    trigram_top_probs = np.zeros((num_ctx, top_k), dtype=np.float32)
    trigram_remaining_mass = np.zeros(num_ctx, dtype=np.float32)

    for out_i, ctx_i in enumerate(tqdm(top_ctx_indices, desc="  Trigram table",
                                       file=sys.stderr)):
        start, end = ctx_starts[ctx_i], ctx_ends[ctx_i]

        these_tokens = next_tokens[start:end]
        these_counts = tri_counts[start:end].astype(np.float64)

        total = these_counts.sum() + TRIGRAM_DELTA * vocab_size
        these_probs = these_counts / total

        k = min(top_k, len(these_tokens))
        if k < len(these_tokens):
            top_indices = np.argpartition(these_probs, -k)[-k:]
            top_indices = top_indices[np.argsort(these_probs[top_indices])[::-1]]
        else:
            top_indices = np.argsort(these_probs)[::-1]

        trigram_context_keys[out_i] = unique_ctx_packed[ctx_i]
        trigram_top_tokens[out_i, :k] = these_tokens[top_indices]
        trigram_top_probs[out_i, :k] = these_probs[top_indices].astype(np.float32)
        trigram_remaining_mass[out_i] = 1.0 - these_probs[top_indices].sum()

    return trigram_context_keys, trigram_top_tokens, trigram_top_probs, trigram_remaining_mass


def save_table(output_path: str, vocab_size: int,
               unigram_probs,
               bigram_context_keys, bigram_top_tokens, bigram_top_probs, bigram_remaining_mass,
               trigram_context_keys, trigram_top_tokens, trigram_top_probs, trigram_remaining_mass,
               tokenizer_name: str):
    """Save all tables to a compressed .npz file."""
    print(f"Saving to {output_path}...", file=sys.stderr)

    np.savez_compressed(
        output_path,
        # Metadata
        vocab_size=np.array([vocab_size], dtype=np.int32),
        tokenizer_name=np.array([tokenizer_name]),
        # Unigram
        unigram_probs=unigram_probs,
        # Bigram
        bigram_context_keys=bigram_context_keys,
        bigram_top_tokens=bigram_top_tokens,
        bigram_top_probs=bigram_top_probs,
        bigram_remaining_mass=bigram_remaining_mass,
        # Trigram
        trigram_context_keys=trigram_context_keys,
        trigram_top_tokens=trigram_top_tokens,
        trigram_top_probs=trigram_top_probs,
        trigram_remaining_mass=trigram_remaining_mass,
    )

    file_size = os.path.getsize(output_path)
    print(f"Table saved: {file_size / 1024 / 1024:.1f} MB", file=sys.stderr)
    print(f"  Unigram entries: {len(unigram_probs):,}", file=sys.stderr)
    print(f"  Bigram contexts: {len(bigram_context_keys):,} "
          f"(top-{bigram_top_tokens.shape[1]})", file=sys.stderr)
    print(f"  Trigram contexts: {len(trigram_context_keys):,} "
          f"(top-{trigram_top_tokens.shape[1]})", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Build precomputed trigram table for compression")
    parser.add_argument("--corpus", choices=["wikitext-2", "wikitext-103"],
                        default=None, help="Download and use a standard corpus")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to a local text file as corpus")
    parser.add_argument("--hf-dataset", type=str, default=None,
                        help="HuggingFace dataset name (e.g., codeparrot/github-code)")
    parser.add_argument("--hf-languages", type=str, default=None,
                        help="Comma-separated language filters (e.g., HTML,CSS,JavaScript,JSON)")
    parser.add_argument("--hf-max-samples", type=int, default=None,
                        help="Max samples to load from HF dataset")
    parser.add_argument("--mix-corpus", type=str, default=None,
                        help="Mix with standard corpus (wikitext-2 or wikitext-103)")
    parser.add_argument("--output", type=str, default="trigram_table.npz",
                        help="Output .npz file path")
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER,
                        help="HuggingFace tokenizer name")
    parser.add_argument("--bigram-top-k", type=int, default=BIGRAM_TOP_K,
                        help="Top-K entries per bigram context")
    parser.add_argument("--trigram-top-k", type=int, default=TRIGRAM_TOP_K,
                        help="Top-K entries per trigram context")
    parser.add_argument("--max-bigram-ctx", type=int, default=MAX_BIGRAM_CONTEXTS,
                        help="Max bigram contexts to keep")
    parser.add_argument("--max-trigram-ctx", type=int, default=MAX_TRIGRAM_CONTEXTS,
                        help="Max trigram contexts to keep")

    args = parser.parse_args()

    if args.corpus is None and args.file is None and args.hf_dataset is None:
        parser.error("Must specify --corpus, --file, or --hf-dataset")

    t0 = time.time()

    # Load corpus
    texts = []

    # Load base corpus
    if args.corpus == "wikitext-2":
        texts.extend(load_corpus_wikitext("wikitext-2-raw-v1"))
    elif args.corpus == "wikitext-103":
        texts.extend(load_corpus_wikitext("wikitext-103-raw-v1"))
    elif args.file:
        texts.extend(load_corpus_file(args.file))

    # Load HuggingFace dataset
    if args.hf_dataset:
        languages = None
        if args.hf_languages:
            languages = [lang.strip() for lang in args.hf_languages.split(",")]
        hf_texts = load_corpus_huggingface(
            args.hf_dataset, languages=languages, max_samples=args.hf_max_samples)
        texts.extend(hf_texts)

    # Optionally mix with another corpus
    if args.mix_corpus == "wikitext-2":
        texts.extend(load_corpus_wikitext("wikitext-2-raw-v1"))
    elif args.mix_corpus == "wikitext-103":
        texts.extend(load_corpus_wikitext("wikitext-103-raw-v1"))

    print(f"\nTotal corpus size: {len(texts)} documents", file=sys.stderr)

    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size:,}", file=sys.stderr)

    # Tokenize
    tokens = tokenize_corpus(texts, tokenizer)

    # Count n-grams
    unigram_counts, (bi_unique, bi_counts), (tri_unique, tri_counts) = \
        count_ngrams(tokens, vocab_size)

    # Build tables
    unigram_probs = build_unigram_table(unigram_counts, UNIGRAM_DELTA)

    bi_keys, bi_tokens, bi_probs, bi_remaining = build_bigram_table(
        bi_unique, bi_counts, unigram_counts, vocab_size,
        args.bigram_top_k, args.max_bigram_ctx)

    tri_keys, tri_tokens, tri_probs, tri_remaining = build_trigram_table(
        tri_unique, tri_counts, vocab_size,
        args.trigram_top_k, args.max_trigram_ctx)

    # Save
    save_table(
        args.output, vocab_size,
        unigram_probs,
        bi_keys, bi_tokens, bi_probs, bi_remaining,
        tri_keys, tri_tokens, tri_probs, tri_remaining,
        args.tokenizer,
    )

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
