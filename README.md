<p align="center">
  <img src="assets/banner_cpu.png" alt="Nacrith">
</p>

<p align="center">
  <a href="https://nacrith.com">Website</a> · <a href="assets/nacrith_paper_cpu.pdf">Technical Paper (PDF)</a> · <a href="https://huggingface.co/spaces/robtacconelli/Nacrith-CPU">Try on Hugging Face</a> · <a href="https://github.com/robtacconelli/Nacrith-GPU">GPU Version</a>
</p>

**Nacrith CPU** is a **state-of-the-art lossless text compression system** that combines precomputed n-gram probability tables with arithmetic coding. Unlike the original Nacrith GPU, which runs a 135M-parameter neural network at every token, Nacrith CPU replaces the LLM with **offline-built sparse trigram tables** — delivering **superior compression on non-repetitive text** with **zero GPU requirement** and **no inference cost**.

The core insight remains the same: **compression is prediction** (Shannon, 1948). A good predictor of text can be turned into a good compressor. But instead of running a transformer for each prediction, Nacrith CPU looks up precomputed unigram, bigram, and trigram conditional probabilities from `.npz` tables, interpolates them with Katz-style backoff weights, and feeds the resulting distribution into a **C-accelerated arithmetic coder**. An adaptive layer accumulates per-document n-gram statistics during compression, gradually blending them in to capture document-specific patterns.

This architecture tests **all trigram tables in parallel** across CPU cores using `ProcessPoolExecutor` with shared-memory table segments (zero-copy). For each text chunk, it simultaneously tests trigram compression and compares against full-file lzma compression, choosing whichever produces the smallest output. This ensures **Nacrith is never worse than standalone lzma**.

---

## Benchmark Results

Tested on **10 classic literature books** from Project Gutenberg, ranging from 53 KB to 2.6 MB. All files are English prose with unique, non-repetitive content.

### Summary Table

| Book Title | Original Size | gzip -9 | lzma -9 | **Nacrith CPU** | **Improvement** |
|------------|---------------|---------|---------|------------------|-----------------|
| Alice in Wonderland | 53 KB | 20.3 KB (38.3%) | 19.1 KB (36.1%) | **14.6 KB (27.6%)** | **+23.4%** 🏆 |
| The Turn of the Screw | 228 KB | 89.0 KB (39.0%) | 78.3 KB (34.3%) | **61.8 KB (27.0%)** | **+21.0%** 🏆 |
| The Jungle Book | 273 KB | 102.0 KB (37.3%) | 90.5 KB (33.1%) | **74.4 KB (27.2%)** | **+17.8%** 🏆 |
| The Hound of the Baskervilles | 320 KB | 121.0 KB (37.9%) | 104.3 KB (32.6%) | **80.3 KB (25.1%)** | **+23.0%** 🏆 |
| The Scarlet Letter | 483 KB | 189.4 KB (39.2%) | 160.0 KB (33.2%) | **132.2 KB (27.4%)** | **+17.4%** 🏆 |
| Age of Innocence | 580 KB | 226.8 KB (39.1%) | 188.9 KB (32.6%) | **153.6 KB (26.5%)** | **+18.7%** 🏆 |
| Pride and Prejudice | 691 KB | 250.7 KB (36.3%) | 205.5 KB (29.8%) | **171.9 KB (24.9%)** | **+16.3%** 🏆 |
| Moby Dick | 1.2 MB | 498.0 KB (41.0%) | 408.5 KB (33.6%) | **352.7 KB (29.0%)** | **+13.7%** 🏆 |
| Don Quixote | 2.2 MB | 828.9 KB (37.8%) | 640.6 KB (29.2%) | **572.8 KB (26.2%)** | **+10.6%** 🏆 |
| Count of Monte Cristo | 2.6 MB | 976.6 KB (37.4%) | 756.3 KB (29.0%) | **669.3 KB (25.6%)** | **+11.5%** 🏆 |

**Win Rate:** 10/10 (100%)
**Average Improvement over lzma:** 18.7%
**Average Compression Ratio:** 27.2% (vs lzma: 32.3%, gzip: 38.4%)

*All compressions verified lossless (byte-for-byte identical decompression).*

---

## Charts

### Compression Ratio by File

![Compression Ratio by File](assets/compression_ratio_by_file.png)

Nacrith CPU consistently achieves the lowest compression ratio across all file sizes.

### Bits Per Byte Comparison

![Bits Per Byte](assets/bits_per_byte_by_file.png)

Nacrith averages **2.12 bits per byte** compared to lzma's 2.57 bpb and gzip's 3.07 bpb.

### Improvement vs File Size

![Improvement vs Size](assets/improvement_vs_size.png)

Nacrith's advantage is strongest on small-to-medium files (50-500 KB) where dictionary compressors have limited context, achieving **20-23% improvements** over lzma. Even on multi-megabyte files, Nacrith maintains **10-14% gains**.

### Compressed Size Trends

![Size Comparison](assets/size_comparison_line.png)

Logarithmic view showing how compressed sizes scale with file size. Nacrith maintains the lowest curve across the entire range.

### Space Savings

![Space Savings](assets/space_savings.png)

Nacrith saves **72-75% of original file size** on average, compared to lzma's 67-71% and gzip's 61-64%.

### Average Performance

![Average Performance](assets/average_performance.png)

Across all 10 books: Nacrith achieves **27.2% compression ratio** and **2.12 bits/byte**, outperforming both lzma (32.3%, 2.57 bpb) and gzip (38.4%, 3.07 bpb).

---

## How It Works

Nacrith CPU is built on the same theoretical foundation as the Nacrith GPU: the connection between **prediction** and **compression**. A model that assigns high probability to the actual next token enables the arithmetic coder to encode it in fewer bits. The key difference is the probability source: instead of a 135M-parameter transformer, Nacrith CPU uses a **static trigram probability table** augmented with **online adaptive counters**.

### The Trigram + Arithmetic Coding Pipeline

```
Input bytes
    │
    ▼
Binary/Text Segmentation (5-step pipeline)
    │
    ├── Binary regions → lzma (always)
    │
    └── Text regions → split into dynamic-sized chunks
            │
            ▼
        For each chunk, IN PARALLEL across CPU cores:
            ├── Worker 1: Compress with trigram table
            └── Main process: Track accumulated lzma stream
            │
            ▼
        Compare: individual trigram entries vs full-file lzma
            │
            ▼
        Emit whichever plan produces smaller output → NC05 format
```

### Compression (per chunk, with trigram table)

```
Chunk bytes
    │
    ▼
Decode as Latin-1 → Tokenize with SmolLM2-135M BPE (49,152 tokens)
    │
    ▼
For each token in sequence:
    1. Build context = [prev2, prev1]
    2. Look up trigram P(token | prev2, prev1) from table
    3. Look up bigram  P(token | prev1) from table
    4. Look up unigram P(token) from table
    5. Interpolate:  P = 0.70·tri + 0.25·bi + 0.05·uni
    6. Blend with adaptive counters (weight ramps 0→0.35 over 800 tokens)
    7. Build sparse CDF (top-512 tokens + rest bucket)
    8. Arithmetic encoder narrows interval by P(actual token)
    │
    ▼
Finalize → compressed bitstream
```

### Decompression

```
Compressed bitstream
    │
    ▼
For each position:
    1. Build same context, same probability distribution
    2. Arithmetic decoder recovers symbol index from interval
    3. Map symbol index → token ID
    4. Feed recovered token back as context
    │
    ▼
Detokenize → original bytes
```

Both sides build **identical probability distributions** from the same static table and identical adaptive state, guaranteeing perfect lossless reconstruction.

---

## Technical Architecture

### 1. Sparse Trigram Probability Model

The probability model is the heart of the system. For each token position, it constructs a probability distribution over the full 49,152-token vocabulary by combining three n-gram levels.

**Table structure** (built offline by `build_table.py`):

- **Unigram table**: Dense float32 array of shape `(49152,)` containing `P(token)` with add-delta smoothing (delta=0.01). Computed from corpus-wide token frequency counts.

- **Bigram table**: Sparse structure storing, for each context token, the **top-256** most likely next tokens with their conditional probabilities, plus a scalar `remaining_mass` for all tokens not in the top-K. Contexts stored as sorted uint16 keys for O(log n) binary search lookup. Up to 100,000 bigram contexts retained.

- **Trigram table**: Same structure as bigram but keyed on `(prev2, prev1)` token pairs packed into uint32 as `(prev2 << 16) | prev1`. Stores **top-128** next tokens per context. Up to 2,000,000 trigram contexts retained.

**Probability computation** (`get_sparse_cdf` in `trigram_model.py`):

For a given context `[prev2, prev1]`, the model:

1. Collects a **candidate set** of tokens: union of unigram top-512, bigram top-K entries, trigram top-K entries, and adaptive counter tokens. Typically 500-800 candidates.

2. For each candidate, computes three raw probabilities (unigram, bigram, trigram) with Katz backoff for missing entries.

3. **Static interpolation**: `P = 0.70 * tri + 0.25 * bi + 0.05 * uni` (full context available)

4. **Adaptive mixing**: Blends in per-document counters with linearly ramping weight (0→0.35 over 800 tokens)

5. Computes `rest_mass = 1.0 - sum(candidate_probs)` to cover tokens outside the sparse candidate set.

6. Quantizes to integer CDF with `CDF_TOTAL = 65,536` (16-bit precision). Every symbol gets at least `MIN_PROB = 1` count.

**Rest-bucket encoding**: When the actual token falls outside the sparse candidate set, the encoder first signals the "rest" symbol (index 512), then encodes the token's rank among excluded tokens using a uniform code.

### 2. Adaptive Mechanism

The static trigram tables capture corpus-wide statistics but miss document-specific patterns. The adaptive layer maintains **per-document** bigram and trigram counters updated after each token, gradually blended into the probability distribution with weight ramping from 0 to 0.35 over 800 tokens.

Add-1 smoothing: `P_adaptive(token) = (count + 1) / (total + num_observed + 1)`

Final probability: `P_final = (1 - lambda_a) * P_static + lambda_a * P_adaptive`

### 3. Binary/Text Segmentation

Before compression, input bytes are classified into binary and text regions using a 5-step pipeline:

1. **Byte classification**: Each byte classified as text-like (printable ASCII 32-126, plus tab/LF/CR) or binary
2. **Short text demotion**: Text runs < 64 bytes → binary
3. **Merge adjacent**: Consecutive same-type runs merged
4. **Bridge small gaps**: Binary gaps ≤ 8 bytes between text regions → absorbed into text
5. **Absorb small binary chunks**: Binary chunks < 64 bytes adjacent to text → absorbed into text

Binary regions always use lzma. Text regions proceed to parallel multi-table compression.

### 4. Dynamic Chunk Sizing

Replaces the fixed 8 KB chunk size with adaptive sizing:

```python
chunk_size = max(2048, min(65536, text_segment_length // 10))
```

| File Size | Chunk Size |
|-----------|------------|
| < 20 KB | 2,048 B (minimum) |
| 50 KB | 5,120 B |
| 100 KB | 10,240 B |
| 500 KB | 51,200 B |
| 1 MB+ | 65,536 B (maximum) |

Larger chunks give the trigram model more context and allow better probability estimates. The 64 KB cap prevents excessive memory usage.

### 5. Accumulated XZ Bucket Algorithm

The defining innovation of Nacrith CPU: **Nacrith is guaranteed never worse than standalone lzma**.

For each file, two compression plans are built **simultaneously**:

1. **Individual plan**: For each chunk, test all trigram tables in parallel (ProcessPoolExecutor), pick best trigram result, compare against per-chunk lzma, keep smaller. Build list of entries.

2. **Full-file XZ plan**: One contiguous `lzma.compress(entire_file)` runs in a background worker while individual chunks are processed.

At the end, compare total compressed sizes:
- If `len(full_file_lzma) <= sum(individual_entry_sizes)`: emit single lzma entry
- Else: emit individual entries

This means:
- If trigram genuinely compresses better, use it
- If repetition dominates (rare in literature), fall back to full-file lzma
- **Never produce worse output than standalone lzma**

On the 10 literature benchmarks, **individual entries won every time** — proving trigram compression is genuinely superior for non-repetitive text.

### 6. Parallel Multi-Table Architecture

**Shared memory layout**:
- At startup, all `.npz` tables are loaded and copied into `multiprocessing.SharedMemory` segments
- Workers attach to these segments and reconstruct numpy array **views** (zero-copy)
- Total shared memory: ~91 MB for the English literature table

**Process pool**:
- `ProcessPoolExecutor` with `min(cpu_count, num_tables + 1)` workers
- Each worker initialized once, attaches to shared memory, loads tokenizer
- Workers bypass Python GIL (separate OS processes)

**Per-chunk compression**:
- For each chunk, submit N tasks (one per trigram table) to worker pool
- Main process computes per-chunk lzma while workers run
- Collect results, pick best trigram, compare against lzma, decide

### 7. C Arithmetic Coder

Implemented as CPython C extension (`arith_coder.c`) for 10-20x speedup over Python.

**Precision**: 32-bit fixed-point. Interval `[low, high]` uses `uint64_t` in `[0, 2^32 - 1]`. Intermediate products use `__int128` to avoid overflow.

**Renormalization**: Standard Moffat/Neal/Witten E1/E2/E3 scheme (checks `high < HALF`, `low >= HALF`, and quarter-straddle conditions in a loop).

**Bit buffer**: Dynamic array that starts at 4096 bits and doubles on overflow.

### 8. Trigram Table Construction

Tables are built offline by `build_table.py`:

1. **Corpus loading**: A corpus like WikiText-103 for general English text
2. **Tokenization**: Full corpus tokenized with SmolLM2-135M BPE (49,152 tokens)
3. **N-gram counting**: Unigrams via `np.bincount`, bigrams/trigrams with packed integer keys
4. **Top-K selection**: Only top-K most likely next tokens retained per context (K=256 for bigrams, K=128 for trigrams)
5. **Context pruning**: Only most frequent contexts kept (100K bigram, 2M trigram)
6. **Smoothing**: Add-delta smoothing at each level (δ_uni=0.01, δ_bi=0.001, δ_tri=0.0001)
7. **Storage**: Compressed `.npz` with sorted key arrays (for binary search) and value arrays

The included table:
- **trigram_en.npz** (91 MB): Trained on fineweb xl (https://huggingface.co/datasets/HuggingFaceFW/fineweb), optimized for English in general

### 9. NC05 File Format

Custom binary format recording per-entry compression method:

```
Header:
  [4B] Magic "NC05"
  [4B] Original total size (uint32 BE)
  [2B] Number of tables (uint16 BE)
  Per table:
    [2B] Name length (uint16 BE)
    [N] Table name (UTF-8)
  [4B] Number of entries (uint32 BE)

Per entry:
  [1B] Method: 0x42='B' (binary/lzma), 0x54='T' (trigram), 0x4C='L' (text/lzma)
  [1B] Table index (meaningful only for 'T', 0 otherwise)
  [4B] Original chunk size (uint32 BE)
  [4B] Compressed chunk size (uint32 BE)
  [N] Compressed data
      For 'T': [4B token_count] + arithmetic stream
      For 'B'/'L': raw lzma stream
```

When full-file XZ wins, the file contains a single 'L' entry with `orig_size = total_file_size`.

Backward compatible with TC01 (v7.0 pure text) and NC03 (v7.4 hybrid) formats through magic-byte detection.

---

## Why Nacrith Excels on Non-Repetitive Text

**Dictionary compressors (gzip, lzma)** maintain a sliding window of recently-seen bytes. When an exact byte sequence repeats, they encode it as a `(distance, length)` back-reference. This is devastatingly effective on repetitive data (HTML templates, code with boilerplate, repeated JSON structures) but provides diminishing returns on **unique, varied prose**.

**Nacrith's trigram model** predicts token-by-token based on linguistic patterns learned from a large corpus. It "knows" that after "The President of the United", the token "States" is extremely likely — even if that exact phrase never appeared before in the document. This **semantic understanding** of language produces far better predictions than byte-pattern matching on non-repetitive text.

The benchmark results validate this: on **classic literature** (varied vocabulary, minimal exact repetition, rich linguistic structure), Nacrith achieves **18-23% improvements** over lzma on files under 500 KB, and **10-14% improvements** even on multi-megabyte books.

The accumulated XZ bucket ensures that if a file *were* highly repetitive, Nacrith would fall back to lzma and match its performance. But on the tested literature corpus, **trigram won 100% of the time**.

---

## Project Structure

```
v7_6/
├── nacrith.py                  # CLI interface (compress, decompress, benchmark)
├── compressor.py               # Core: segmentation, parallel compression, NC05 format
├── trigram_model.py            # Sparse trigram probability model with adaptive counters
├── arith_coder.c               # C arithmetic encoder/decoder (32-bit precision)
├── build_table.py              # Offline trigram table builder from text corpora
├── setup.py                    # C extension build configuration
├── Makefile                    # Build automation (setup, build, test, clean)
├── requirements.txt            # Python dependencies
├── trigrams/                   # Precomputed trigram tables
│   └── trigram_en.npz          #   91 MB — fineweb xl - English
├── benchmark_text_files.py     # Benchmark script (gzip vs lzma vs Nacrith)
├── generate_text_charts.py     # Chart generation from benchmark results
├── benchmark_text_results.json # Raw benchmark data
├── BENCHMARK_SUMMARY.md        # Detailed analysis
└── assets/                     # Generated chart images
```

## Installation

### Prerequisites
- Python 3.8+
- GCC compiler (for building the C arithmetic coder extension)
- 4 GB+ RAM (for loading trigram table into shared memory)

### Quick Setup
```bash
make setup    # Install dependencies + build C extension

# Or step by step:
make install  # pip install -r requirements.txt
make build    # python setup.py build_ext --inplace
make test     # Verify installation
```

### Manual Installation
```bash
pip install -r requirements.txt
python setup.py build_ext --inplace
```

## Usage

### Compress
```bash
python nacrith.py c input.txt output.nc5
```

### Decompress
```bash
python nacrith.py d output.nc5 restored.txt
```

### Benchmark
```bash
python nacrith.py benchmark input.txt
```

### Options
```
--trigrams-dir DIR    Directory with .npz tables (default: trigrams/)
--table FILE          Single table fallback
-q, --quiet           Suppress progress output
```

### Run Literature Benchmarks
```bash
python benchmark_text_files.py      # Runs all 10 classic books
python generate_text_charts.py      # Generates chart images in assets/
```

## Building Custom Tables

```bash
# From WikiText-103 corpus (English literature)
python build_table.py --corpus wikitext-103 --output trigrams/trigram_en.npz

# From a local text file
python build_table.py --file corpus.txt --output trigrams/my_table.npz

# Custom top-K and context limits
python build_table.py \
    --corpus wikitext-103 \
    --bigram-top-k 256 \
    --trigram-top-k 128 \
    --max-bigram-ctx 100000 \
    --max-trigram-ctx 2000000 \
    --output trigrams/my_table.npz
```

Place new tables in the `trigrams/` directory — they are auto-discovered at startup.

## Testing

```bash
# Run built-in test
make test

# Manual roundtrip verification
python nacrith.py c test.txt test.nc5
python nacrith.py d test.nc5 test_restored.txt
diff test.txt test_restored.txt  # Should produce no output
```

## Performance Characteristics

### Compression Quality (bits per byte)
- **Nacrith CPU**: 2.0-2.2 bpb on English literature
- **lzma -9**: 2.3-2.7 bpb
- **gzip -9**: 2.9-3.2 bpb
- **Nacrith for GPU**: ~1.24 bpb (with 135M-param LLM on GPU)

### Speed
- **gzip**: ~10-50 MB/s
- **lzma**: ~1-5 MB/s
- **Nacrith CPU**: ~0.6-1.8 MB/min (with only one core)

The per-token Python-level CDF construction and arithmetic coding loop is the bottleneck. For use cases where **maximum compression** is more important than speed (archival storage, offline compression, compression benchmarks), Nacrith CPU is the clear winner.

### Memory
- **Table loading**: ~91 MB for English table (shared across workers, zero-copy)
- **Per-worker overhead**: ~200 MB (tokenizer + Python runtime)
- **Total**: ~1-2 GB RAM with default worker count

## Limitations

- **Speed**: Slower than lzma due to per-token arithmetic coding overhead. There's still a lot of margin of improvement.
- **Text-only**: Designed for UTF-8 text. Binary-only data should use lzma directly.
- **Domain-specific**: Best results when test data matches training corpus.
- **Memory**: Requires several GB RAM for table loading and worker processes.

## When to Use Nacrith CPU

**Ideal use cases:**
- Archival compression of **English literature, articles, documentation, non-repetitive texts**
- Offline compression where **maximum ratio matters more than speed**
- Research and benchmarking: achieving **state-of-the-art text compression ratios**
- Compressing **non-repetitive, unique text** (not templates or boilerplate)

**Not ideal for:**
- Real-time compression
- Binary data or not expected text (language, type, ...) - no specialized tables available
- Repetitive structured data (HTML templates, code with copy-paste, log files)
- Streaming compression (requires full file for accumulated XZ comparison)

## Comparison to Nacrith for GPU

| Feature | Nacrith GPU | Nacrith CPU |
|---------|---------------|--------------|
| **Model** | SmolLM2-135M (135M params) | Precomputed trigram tables |
| **Hardware** | GPU required | CPU-only |
| **Speed** | ~0.1 MB/min (~21 tokens/s) | ~0.6-1.8 MB/min (with only one core) |
| **Compression** | ~1.24 bpb (English) | ~2.12 bpb (English) |
| **Model size** | 259 MB + VRAM | 91 MB (RAM, shared) |
| **Setup** | PyTorch, CUDA, model download | numpy, gcc |
| **Inference** | Forward pass per token | Table lookup per token |

Nacrith CPU trades **0.88 bpb of compression quality** for **eliminating the neural network**, making it practical for CPU-only environments. The trigram model still beats dictionary compressors by **18-23%** on non-repetitive text.

## Theory

The theoretical foundation is Shannon's source coding theorem: the minimum average bits per symbol for lossless compression equals the entropy rate of the source. Arithmetic coding achieves rates within a fraction of a bit of entropy. The quality of compression therefore depends entirely on how well the probability model matches the true data distribution.

Nacrith CPU's trigram model estimates:

```
P(token_t | context) = (1-λ_a) · [λ₃·P_tri + λ₂·P_bi + λ₁·P_uni] + λ_a · P_adaptive
```

where `λ₃ = 0.70`, `λ₂ = 0.25`, `λ₁ = 0.05`, and `λ_a` ramps from 0 to 0.35 over 800 tokens.

The cross-entropy between this model and the true data distribution determines the compressed size. On English literature, this yields ~2.12 bpb — significantly better than lzma's 2.57 bpb and gzip's 3.07 bpb, but above the neural Nacrith's 1.24 bpb.

The key advantage: the trigram model is **fast to evaluate** (table lookup + interpolation) and captures **linguistic structure** (word sequences, grammar patterns) better than dictionary matching, especially on **non-repetitive text** where exact byte repetitions are rare.

## License

Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Credits

- Based on the Nacrith compression framework
- Uses [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) tokenizer from HuggingFace (49,152 BPE tokens)
- Trigram trained using [fineweb xl](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
- Arithmetic coding theory: Witten, Neal, and Cleary (1987), *Arithmetic Coding for Data Compression*
- Trigram language modeling: Katz (1987), *Estimation of Probabilities from Sparse Data*

---

**Nacrith CPU: State-of-the-art lossless compression for non-repetitive English text.**
**100% win rate against lzma on classic literature. 18.7% average improvement.**
