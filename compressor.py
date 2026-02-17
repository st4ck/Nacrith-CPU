"""
Hybrid compressor (v7.6): TRUE PARALLEL multiprocessing + accumulated XZ bucket.

1. Auto-discover all trigram tables in trigrams/ directory
2. Load all numpy arrays into shared memory ONCE (zero-copy for workers)
3. Dynamic chunk sizing: max(2048, min(65536, segment_len // 10))
4. Full-file contiguous XZ: one lzma.compress(entire file) runs in a
   background worker while per-chunk trigram/lzma entries are computed.
   At the end, emit whichever plan (full-file XZ vs individual entries)
   produces smaller output.
5. Trigram tables tested in PARALLEL via ProcessPoolExecutor

Key improvements over v7.5:
 - ProcessPoolExecutor bypasses the GIL for true CPU parallelism
 - Shared memory segments avoid duplicating 3.9GB of tables
 - All CPU cores utilized (workers = min(cpu_count, num_tables + 1))
 - Full-file contiguous XZ exploits cross-chunk repetition for better ratios
 - Dynamic chunk sizing adapts to file size

File formats:
  TC01 -- pure text (single stream, backward compat)
  NC03 -- hybrid chunked format (binary + text sub-chunks, single table)
  NC05 -- parallel multi-table format (adds table_id per trigram entry)
"""
import glob
import lzma
import os
import struct
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory, cpu_count

import numpy as np
from transformers import AutoTokenizer

# Imports needed in main process for decompression
from arithmetic_coder import ArithmeticEncoder, ArithmeticDecoder

# ---- format constants ----
MAGIC_TEXT  = b'TC01'
MAGIC_NC03  = b'NC03'
MAGIC_CHUNK = b'NC05'

METHOD_BINARY  = 0x42  # 'B' -- binary, always lzma
METHOD_TRIGRAM = 0x54  # 'T' -- text sub-chunk, trigram won
METHOD_LZMA   = 0x4C  # 'L' -- text sub-chunk, lzma won

# ---- chunk sizes ----
# Dynamic chunk sizing: computed per text segment in compress_bytes()
# chunk_size = max(2048, min(65536, segment_len // 10))

# ---- binary/text detection thresholds ----
MIN_TEXT_RUN     = 64
MAX_BRIDGE_GAP   = 8
MIN_BINARY_CHUNK = 64

CHUNK_TYPE_TEXT   = 0x54  # internal, for segmentation
CHUNK_TYPE_BINARY = 0x42

# Bytes considered text-like: printable ASCII + tab/LF/CR
TEXT_BYTES = frozenset(range(32, 127)) | {9, 10, 13}


# ==================================================================
# Binary/text segmentation (same as v7.4/v7.5)
# ==================================================================

def _segment_chunks(data: bytes) -> list[tuple[int, int, int]]:
    """Segment data into text and binary regions.

    Returns list of (chunk_type, offset, length).
    """
    if not data:
        return []

    # Step 1: classify contiguous runs
    runs = []
    current_type = CHUNK_TYPE_TEXT if data[0] in TEXT_BYTES else CHUNK_TYPE_BINARY
    run_start = 0
    for i in range(1, len(data)):
        byte_type = CHUNK_TYPE_TEXT if data[i] in TEXT_BYTES else CHUNK_TYPE_BINARY
        if byte_type != current_type:
            runs.append((current_type, run_start, i - run_start))
            current_type = byte_type
            run_start = i
    runs.append((current_type, run_start, len(data) - run_start))

    # Step 2: demote short text runs to binary
    runs = [
        (CHUNK_TYPE_BINARY if t == CHUNK_TYPE_TEXT and length < MIN_TEXT_RUN else t,
         off, length)
        for t, off, length in runs
    ]

    # Step 3: merge adjacent same-type
    runs = _merge_adjacent(runs)

    # Step 4: bridge small binary gaps between text runs
    if len(runs) >= 3:
        bridged = [runs[0]]
        i = 1
        while i < len(runs) - 1:
            prev_t = bridged[-1][0]
            curr_t, curr_off, curr_len = runs[i]
            next_t = runs[i + 1][0]
            if (prev_t == CHUNK_TYPE_TEXT and curr_t == CHUNK_TYPE_BINARY
                    and next_t == CHUNK_TYPE_TEXT and curr_len <= MAX_BRIDGE_GAP):
                prev_t2, prev_off, prev_len = bridged[-1]
                _, _, next_len = runs[i + 1]
                bridged[-1] = (CHUNK_TYPE_TEXT, prev_off,
                               prev_len + curr_len + next_len)
                i += 2
            else:
                bridged.append((curr_t, curr_off, curr_len))
                i += 1
        if i < len(runs):
            bridged.append(runs[i])
        runs = bridged

    # Step 5: merge again
    runs = _merge_adjacent(runs)

    # Step 6: absorb small binary chunks into adjacent text
    if len(runs) >= 2:
        absorbed = []
        i = 0
        while i < len(runs):
            t, off, length = runs[i]
            if t == CHUNK_TYPE_BINARY and length < MIN_BINARY_CHUNK:
                left_text = (absorbed and absorbed[-1][0] == CHUNK_TYPE_TEXT)
                right_text = (i + 1 < len(runs)
                              and runs[i + 1][0] == CHUNK_TYPE_TEXT)
                if left_text and right_text:
                    prev_t, prev_off, prev_len = absorbed[-1]
                    _, _, next_len = runs[i + 1]
                    absorbed[-1] = (CHUNK_TYPE_TEXT, prev_off,
                                    prev_len + length + next_len)
                    i += 2
                    continue
                elif left_text:
                    prev_t, prev_off, prev_len = absorbed[-1]
                    absorbed[-1] = (CHUNK_TYPE_TEXT, prev_off,
                                    prev_len + length)
                    i += 1
                    continue
                elif right_text:
                    absorbed.append((CHUNK_TYPE_TEXT, off, length))
                    i += 1
                    continue
            absorbed.append((t, off, length))
            i += 1
        runs = _merge_adjacent(absorbed)

    return runs


def _merge_adjacent(runs):
    if not runs:
        return runs
    merged = [runs[0]]
    for t, off, length in runs[1:]:
        if t == merged[-1][0]:
            prev_t, prev_off, prev_len = merged[-1]
            merged[-1] = (prev_t, prev_off, prev_len + length)
        else:
            merged.append((t, off, length))
    return merged


# ==================================================================
# Text compression helpers
# ==================================================================

def _bisect(sorted_arr, val):
    lo, hi = 0, len(sorted_arr) - 1
    while lo <= hi:
        mid = (lo + hi) >> 1
        v = int(sorted_arr[mid])
        if v == val:
            return mid
        elif v < val:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def _encode_uniform(encoder, value, total):
    if total <= 1:
        return
    if total <= 16384:
        cdf = list(range(total + 1))
        encoder.encode_symbol(cdf, value)
    else:
        hi_total = (total + 255) // 256
        hi_val = value // 256
        lo_total = min(256, total - hi_val * 256)
        lo_val = value % 256
        if lo_val >= lo_total:
            lo_val = lo_total - 1
        cdf_hi = list(range(hi_total + 1))
        encoder.encode_symbol(cdf_hi, hi_val)
        cdf_lo = list(range(lo_total + 1))
        encoder.encode_symbol(cdf_lo, lo_val)


def _decode_uniform(decoder, total):
    if total <= 1:
        return 0
    if total <= 16384:
        cdf = list(range(total + 1))
        return decoder.decode_symbol(cdf)
    else:
        hi_total = (total + 255) // 256
        cdf_hi = list(range(hi_total + 1))
        hi_val = decoder.decode_symbol(cdf_hi)
        lo_total = min(256, total - hi_val * 256)
        cdf_lo = list(range(lo_total + 1))
        lo_val = decoder.decode_symbol(cdf_lo)
        return hi_val * 256 + lo_val


def _rank_to_token(rank, excluded_sorted):
    token_id = 0
    remaining = rank
    ex_idx = 0
    n_ex = len(excluded_sorted)
    while True:
        while ex_idx < n_ex and excluded_sorted[ex_idx] == token_id:
            token_id += 1
            ex_idx += 1
        if remaining == 0:
            return token_id
        remaining -= 1
        token_id += 1


# ==================================================================
# Standalone trigram compression/decompression functions
# (used by main process for decompression and TC01 backward compat)
# ==================================================================

def _trigram_compress_chunk(model, tokenizer, chunk_bytes):
    """Compress one chunk with a given trigram model.

    Returns (num_tokens, stream_bytes).
    """
    text = chunk_bytes.decode('latin-1')
    token_ids = tokenizer.encode(text)
    num_tokens = len(token_ids)
    if num_tokens == 0:
        return 0, b''
    model.reset()
    encoder = ArithmeticEncoder()
    context = []

    for token_id in token_ids:
        sparse_tokens, cdf = model.get_sparse_cdf(context)
        excluded_sorted = sorted(sparse_tokens.tolist())
        pos = _bisect(sparse_tokens, token_id)

        if pos >= 0:
            encoder.encode_symbol(cdf, pos)
        else:
            rest_idx = len(sparse_tokens)
            encoder.encode_symbol(cdf, rest_idx)
            rest_size, rank = model.get_rest_rank(token_id, excluded_sorted)
            _encode_uniform(encoder, rank, rest_size)

        model.update(token_id)
        context.append(token_id)

    stream = encoder.finish()
    return num_tokens, stream


def _trigram_decompress_chunk(model, tokenizer, stream, num_tokens):
    """Decompress a trigram stream back to bytes using a given model."""
    if num_tokens == 0:
        return b''
    model.reset()
    decoder = ArithmeticDecoder(stream)
    context = []
    token_ids = []

    for i in range(num_tokens):
        sparse_tokens, cdf = model.get_sparse_cdf(context)
        excluded_sorted = sorted(sparse_tokens.tolist())
        sym = decoder.decode_symbol(cdf)

        if sym < len(sparse_tokens):
            token_id = int(sparse_tokens[sym])
        else:
            rest_size = model.vocab_size - len(excluded_sorted)
            if rest_size <= 0:
                rest_size = 1
            rank = _decode_uniform(decoder, rest_size)
            token_id = _rank_to_token(rank, excluded_sorted)

        token_ids.append(token_id)
        model.update(token_id)
        context.append(token_id)

    text = tokenizer.decode(token_ids)
    return text.encode('latin-1')


# ==================================================================
# Table discovery
# ==================================================================

def discover_trigram_tables(trigrams_dir):
    """Auto-discover all .npz trigram tables in the given directory.

    Returns list of (table_name, table_path) sorted by name.
    """
    if not os.path.isdir(trigrams_dir):
        return []

    tables = []
    for path in sorted(glob.glob(os.path.join(trigrams_dir, "*.npz"))):
        name = os.path.splitext(os.path.basename(path))[0]
        # Remove common prefixes for cleaner display
        display_name = name
        if display_name.startswith("trigram_"):
            display_name = display_name[8:]
        tables.append((display_name, os.path.abspath(path)))

    return tables


# ==================================================================
# Shared memory management for trigram tables
# ==================================================================

# Model constants (imported from trigram_model to keep behavior identical)
from trigram_model import (
    CDF_TOTAL, MIN_PROB, LAMBDA_TRI, LAMBDA_BI, LAMBDA_UNI,
    LAMBDA_BI_ONLY, LAMBDA_UNI_ONLY, MAX_ADAPTIVE_WEIGHT,
    ADAPTIVE_RAMP_TOKENS, SPARSE_TOP_K,
)


def _create_shm_for_array(arr):
    """Create a shared memory segment and copy a numpy array into it.

    Returns (shm_name, dtype_str, shape_tuple, shm_object).
    """
    nbytes = arr.nbytes
    shm = shared_memory.SharedMemory(create=True, size=max(nbytes, 1))
    # Copy array data into shared memory buffer
    shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    shm_arr[:] = arr[:]
    return shm.name, str(arr.dtype), arr.shape, shm


def _load_table_to_shared_memory(table_path, table_name, verbose=True):
    """Load one .npz trigram table into shared memory segments.

    Memory-efficient: loads each array, copies to shared memory, then
    immediately frees the original numpy array to minimize peak usage.

    Returns (shm_info_dict, list_of_shm_objects).
    shm_info_dict contains everything workers need to reconstruct the table.
    """
    import gc

    if verbose:
        print(f"  Loading {table_name} into shared memory...",
              file=sys.stderr)

    data = np.load(table_path, allow_pickle=True)
    vocab_size = int(data["vocab_size"][0])
    tokenizer_name = str(data["tokenizer_name"][0])

    shm_objects = []
    shm_info = {
        "vocab_size": vocab_size,
        "tokenizer_name": tokenizer_name,
        "table_name": table_name,
        "arrays": {},
    }

    # Define array name -> (npz key, target dtype or None for original)
    # Keep original dtypes (float32) to minimize memory usage.
    # Only unigram_probs is promoted to float64 (it's tiny: 49K entries).
    # All other arrays stay in their native dtype from the .npz file.
    # The float32->float64 conversion for probability computation happens
    # on-the-fly during get_sparse_cdf (only for the small per-chunk slices).
    array_specs = [
        ("unigram_probs",  "unigram_probs",          np.float64),
        ("bi_ctx_keys",    "bigram_context_keys",     None),
        ("bi_top_tokens",  "bigram_top_tokens",       None),
        ("bi_top_probs",   "bigram_top_probs",        None),
        ("bi_remaining",   "bigram_remaining_mass",   None),
        ("tri_ctx_keys",   "trigram_context_keys",    None),
        ("tri_top_tokens", "trigram_top_tokens",       None),
        ("tri_top_probs",  "trigram_top_probs",        None),
        ("tri_remaining",  "trigram_remaining_mass",   None),
    ]

    # Load each array one at a time, copy to shm, free original
    uni_probs_shm_name = None
    for arr_name, npz_key, target_dtype in array_specs:
        arr = data[npz_key]
        if target_dtype is not None:
            arr = arr.astype(target_dtype)
        name, dtype_str, shape, shm_obj = _create_shm_for_array(arr)
        shm_info["arrays"][arr_name] = {
            "shm_name": name,
            "dtype": dtype_str,
            "shape": shape,
        }
        shm_objects.append(shm_obj)
        # Remember unigram shm for top-K computation
        if arr_name == "unigram_probs":
            uni_probs_shm_name = name
            uni_probs_shape = shape
            uni_probs_dtype = dtype_str
        del arr
        gc.collect()

    # Compute unigram top-K indices from the shared memory copy
    # (avoids keeping the original in regular memory)
    uni_shm = shared_memory.SharedMemory(
        name=uni_probs_shm_name, create=False)
    uni_probs = np.ndarray(
        uni_probs_shape, dtype=np.dtype(uni_probs_dtype),
        buffer=uni_shm.buf)
    uni_top_idx = np.argsort(uni_probs)[::-1][:SPARSE_TOP_K].copy()
    uni_top_idx = uni_top_idx.astype(np.int64)
    uni_shm.close()

    name, dtype_str, shape, shm_obj = _create_shm_for_array(uni_top_idx)
    shm_info["arrays"]["uni_top_idx"] = {
        "shm_name": name,
        "dtype": dtype_str,
        "shape": shape,
    }
    shm_objects.append(shm_obj)

    del uni_top_idx, data
    gc.collect()

    return shm_info, shm_objects


# ==================================================================
# Worker process: global state and initialization
# ==================================================================

# Global state in each worker process (set by _worker_init)
_worker_tables = None       # list of reconstructed table dicts
_worker_tokenizer = None    # tokenizer instance for this worker
_worker_shm_refs = None     # SharedMemory refs (keep alive in worker)


def _worker_init(all_table_shm_info, tokenizer_name):
    """Initialize a worker process.

    Attach to shared memory segments, reconstruct numpy array views
    (zero-copy), and load the tokenizer. Called once per worker process.
    """
    global _worker_tables, _worker_tokenizer, _worker_shm_refs

    _worker_shm_refs = []
    _worker_tables = []

    for tinfo in all_table_shm_info:
        table = {
            "vocab_size": tinfo["vocab_size"],
            "tokenizer_name": tinfo["tokenizer_name"],
            "table_name": tinfo["table_name"],
        }

        # Reconstruct numpy arrays from shared memory (zero-copy views)
        for arr_name, arr_info in tinfo["arrays"].items():
            shm = shared_memory.SharedMemory(
                name=arr_info["shm_name"], create=False)
            _worker_shm_refs.append(shm)
            arr = np.ndarray(
                arr_info["shape"],
                dtype=np.dtype(arr_info["dtype"]),
                buffer=shm.buf,
            )
            table[arr_name] = arr

        # Build unigram top set from shared uni_top_idx
        table["uni_top_set"] = set(table["uni_top_idx"].tolist())

        _worker_tables.append(table)

    # Load tokenizer (lightweight, each worker gets its own)
    _worker_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)


# ==================================================================
# Worker-side adaptive trigram model (stateful per chunk)
# ==================================================================

class _WorkerAdaptiveModel:
    """Lightweight adaptive trigram model for worker processes.

    Uses shared-memory numpy arrays (zero-copy read-only views) for
    the static trigram tables, with per-chunk adaptive counters.
    Produces identical output to AdaptiveTrigramModel from trigram_model.py.
    """

    def __init__(self, table_dict):
        self.vocab_size = table_dict["vocab_size"]
        self.tokenizer_name = table_dict["tokenizer_name"]

        # Shared-memory arrays (read-only views, no copy)
        self.unigram_probs = table_dict["unigram_probs"]
        self.bi_ctx_keys = table_dict["bi_ctx_keys"]
        self.bi_top_tokens = table_dict["bi_top_tokens"]
        self.bi_top_probs = table_dict["bi_top_probs"]
        self.bi_remaining = table_dict["bi_remaining"]
        self.tri_ctx_keys = table_dict["tri_ctx_keys"]
        self.tri_top_tokens = table_dict["tri_top_tokens"]
        self.tri_top_probs = table_dict["tri_top_probs"]
        self.tri_remaining = table_dict["tri_remaining"]
        self._uni_top_idx = table_dict["uni_top_idx"]
        self._uni_top_set = table_dict["uni_top_set"]

        self.reset()

    def reset(self):
        self.adapt_bi = defaultdict(Counter)
        self.adapt_tri = defaultdict(Counter)
        self.tokens_seen = 0
        self._prev1 = None
        self._prev2 = None

    def update(self, token_id):
        if self._prev1 is not None:
            self.adapt_bi[self._prev1][token_id] += 1
        if self._prev2 is not None and self._prev1 is not None:
            self.adapt_tri[(self._prev2, self._prev1)][token_id] += 1
        self._prev2 = self._prev1
        self._prev1 = token_id
        self.tokens_seen += 1

    def _lookup_bigram(self, prev1):
        idx = np.searchsorted(self.bi_ctx_keys, prev1)
        if idx < len(self.bi_ctx_keys) and self.bi_ctx_keys[idx] == prev1:
            # Cast small per-context slices to float64 for precision
            return (self.bi_top_tokens[idx],
                    self.bi_top_probs[idx].astype(np.float64),
                    float(self.bi_remaining[idx]))
        return None

    def _lookup_trigram(self, prev2, prev1):
        key = np.uint32((prev2 & 0xFFFF) << 16 | (prev1 & 0xFFFF))
        idx = np.searchsorted(self.tri_ctx_keys, key)
        if idx < len(self.tri_ctx_keys) and self.tri_ctx_keys[idx] == key:
            # Cast small per-context slices to float64 for precision
            return (self.tri_top_tokens[idx],
                    self.tri_top_probs[idx].astype(np.float64),
                    float(self.tri_remaining[idx]))
        return None

    def get_sparse_cdf(self, context):
        """Return (token_ids, cdf) -- identical semantics to
        AdaptiveTrigramModel.get_sparse_cdf.
        """
        # --- Step 1: Collect candidate token set ---
        candidates = set(self._uni_top_set)

        bi_result = None
        tri_result = None

        if len(context) >= 1:
            prev1 = context[-1]
            bi_result = self._lookup_bigram(prev1)
            if bi_result is not None:
                valid = bi_result[1] > 0
                candidates.update(bi_result[0][valid].tolist())

            if len(context) >= 2:
                prev2 = context[-2]
                tri_result = self._lookup_trigram(prev2, prev1)
                if tri_result is not None:
                    valid = tri_result[1] > 0
                    candidates.update(tri_result[0][valid].tolist())

        # Adaptive tokens
        lambda_a = min(MAX_ADAPTIVE_WEIGHT,
                       self.tokens_seen / ADAPTIVE_RAMP_TOKENS)
        ada_bi_counter = None
        ada_tri_counter = None
        if lambda_a > 1e-12 and len(context) >= 1:
            prev1 = context[-1]
            ada_bi_counter = self.adapt_bi.get(prev1)
            if ada_bi_counter:
                candidates.update(ada_bi_counter.keys())
            if len(context) >= 2:
                prev2 = context[-2]
                ada_tri_counter = self.adapt_tri.get((prev2, prev1))
                if ada_tri_counter:
                    candidates.update(ada_tri_counter.keys())

        token_ids = np.array(sorted(candidates), dtype=np.int64)
        n = len(token_ids)

        # --- Step 2: Build probability for each candidate ---
        uni_probs = self.unigram_probs[token_ids]

        if (len(context) >= 2 and tri_result is not None
                and bi_result is not None):
            bi_probs = uni_probs * bi_result[2]
            bi_tok = bi_result[0]
            bi_p = bi_result[1]
            bi_valid = bi_p > 0
            if bi_valid.any():
                _map_into(bi_probs, token_ids, bi_tok[bi_valid],
                          bi_p[bi_valid])

            tri_probs = bi_probs * tri_result[2]
            tri_tok = tri_result[0]
            tri_p = tri_result[1]
            tri_valid = tri_p > 0
            if tri_valid.any():
                _map_into(tri_probs, token_ids, tri_tok[tri_valid],
                          tri_p[tri_valid])

            static_probs = (LAMBDA_TRI * tri_probs + LAMBDA_BI * bi_probs
                            + LAMBDA_UNI * uni_probs)

        elif len(context) >= 2 and tri_result is not None:
            tri_probs = uni_probs * tri_result[2]
            tri_tok = tri_result[0]
            tri_p = tri_result[1]
            tri_valid = tri_p > 0
            if tri_valid.any():
                _map_into(tri_probs, token_ids, tri_tok[tri_valid],
                          tri_p[tri_valid])
            static_probs = ((LAMBDA_TRI + LAMBDA_BI) * tri_probs
                            + LAMBDA_UNI * uni_probs)

        elif bi_result is not None:
            bi_probs = uni_probs * bi_result[2]
            bi_tok = bi_result[0]
            bi_p = bi_result[1]
            bi_valid = bi_p > 0
            if bi_valid.any():
                _map_into(bi_probs, token_ids, bi_tok[bi_valid],
                          bi_p[bi_valid])
            static_probs = (LAMBDA_BI_ONLY * bi_probs
                            + LAMBDA_UNI_ONLY * uni_probs)

        else:
            static_probs = uni_probs.copy()

        # --- Step 3: Adaptive mixing ---
        if lambda_a > 1e-12 and (ada_bi_counter or ada_tri_counter):
            ada_bi_dist = None
            if ada_bi_counter:
                ada_bi_dist = _build_adaptive_sparse(
                    ada_bi_counter, token_ids, static_probs)
            ada_tri_dist = None
            if ada_tri_counter:
                ada_tri_dist = _build_adaptive_sparse(
                    ada_tri_counter, token_ids, static_probs)

            if ada_tri_dist is not None and ada_bi_dist is not None:
                adaptive = 0.6 * ada_tri_dist + 0.4 * ada_bi_dist
            elif ada_tri_dist is not None:
                adaptive = ada_tri_dist
            else:
                adaptive = ada_bi_dist

            final_probs = ((1.0 - lambda_a) * static_probs
                           + lambda_a * adaptive)
        else:
            final_probs = static_probs

        # --- Step 4: Rest mass ---
        final_probs = np.maximum(final_probs, 1e-10)
        candidate_sum = final_probs.sum()
        rest_mass = max(1e-10, 1.0 - candidate_sum)

        # --- Step 5: Build integer CDF ---
        total_symbols = n + 1
        usable = CDF_TOTAL - total_symbols * MIN_PROB

        all_probs = np.empty(total_symbols, dtype=np.float64)
        all_probs[:n] = final_probs
        all_probs[n] = rest_mass
        all_probs /= all_probs.sum()

        counts = (all_probs * usable).astype(np.int64)
        counts = np.maximum(counts, 0) + MIN_PROB
        diff = CDF_TOTAL - counts.sum()
        if diff != 0:
            counts[counts.argmax()] += diff

        cdf = np.zeros(total_symbols + 1, dtype=np.int64)
        np.cumsum(counts, out=cdf[1:])
        cdf[-1] = CDF_TOTAL

        return token_ids, cdf.tolist()

    def get_rest_rank(self, token_id, excluded_sorted):
        rest_size = self.vocab_size - len(excluded_sorted)
        if rest_size <= 0:
            rest_size = 1
        lo, hi = 0, len(excluded_sorted)
        while lo < hi:
            mid = (lo + hi) >> 1
            if excluded_sorted[mid] < token_id:
                lo = mid + 1
            else:
                hi = mid
        rank = token_id - lo
        return rest_size, rank


def _map_into(target, target_tokens, src_tokens, src_probs):
    """Set target[i] = src_probs[j] where target_tokens[i] == src_tokens[j]."""
    idx = np.searchsorted(target_tokens, src_tokens)
    valid = (idx < len(target_tokens)) & (target_tokens[idx] == src_tokens)
    target[idx[valid]] = src_probs[valid]


def _build_adaptive_sparse(counter, token_ids, static_probs):
    """Build adaptive distribution over sparse token_ids from a Counter."""
    n = len(token_ids)
    dist = static_probs.copy()

    if not counter:
        return dist

    obs_tokens = np.array(list(counter.keys()), dtype=np.int64)
    obs_counts = np.array(list(counter.values()), dtype=np.float64)
    total = obs_counts.sum()
    denom = total + len(obs_tokens) + 1.0
    smoothed = (obs_counts + 1.0) / denom
    remaining_frac = 1.0 / denom

    dist *= remaining_frac

    idx = np.searchsorted(token_ids, obs_tokens)
    valid = (idx < n) & (token_ids[idx] == obs_tokens)
    dist[idx[valid]] = smoothed[valid]

    s = dist.sum()
    if s > 0:
        dist /= s
    return dist


# ==================================================================
# Worker-side compression functions (run in child processes)
# ==================================================================

def _worker_compress_with_table(table_idx, chunk_bytes):
    """Compress chunk_bytes using trigram table[table_idx].

    Runs in a worker process. Uses global _worker_tables and
    _worker_tokenizer initialized by _worker_init.

    Returns (table_idx, METHOD_TRIGRAM, compressed_data) or
            (table_idx, None, None) on failure.
    """
    global _worker_tables, _worker_tokenizer

    try:
        table_dict = _worker_tables[table_idx]
        model = _WorkerAdaptiveModel(table_dict)

        text = chunk_bytes.decode('latin-1')
        token_ids_list = _worker_tokenizer.encode(text)
        num_tokens = len(token_ids_list)
        if num_tokens == 0:
            return (table_idx, METHOD_TRIGRAM, struct.pack('>I', 0))

        model.reset()
        encoder = ArithmeticEncoder()
        context = []

        for token_id in token_ids_list:
            sparse_tokens, cdf = model.get_sparse_cdf(context)
            excluded_sorted = sorted(sparse_tokens.tolist())
            pos = _bisect(sparse_tokens, token_id)

            if pos >= 0:
                encoder.encode_symbol(cdf, pos)
            else:
                rest_idx = len(sparse_tokens)
                encoder.encode_symbol(cdf, rest_idx)
                rest_size, rank = model.get_rest_rank(
                    token_id, excluded_sorted)
                _encode_uniform(encoder, rank, rest_size)

            model.update(token_id)
            context.append(token_id)

        stream = encoder.finish()
        tri_data = struct.pack('>I', num_tokens) + stream
        return (table_idx, METHOD_TRIGRAM, tri_data)

    except Exception:
        return (table_idx, None, None)


def _worker_compress_with_lzma(chunk_bytes):
    """Compress chunk_bytes using lzma.

    Runs in a worker process.
    Returns (-1, METHOD_LZMA, compressed_data).
    """
    return (-1, METHOD_LZMA, lzma.compress(chunk_bytes))


# ==================================================================
# Main compressor class (v7.6: true parallel multiprocessing)
# ==================================================================

class TrigramCompressor:
    """Multi-table compressor with TRUE PARALLEL multiprocessing.

    Architecture:
    1. Loads all trigram tables into shared memory (loaded ONCE, zero-copy)
    2. Creates ProcessPoolExecutor with N workers (bypasses GIL)
       N = min(cpu_count, num_tables + 1)
    3. Two competing plans built simultaneously:
       a) Individual plan: per-chunk best of trigram/lzma (dynamic chunk sizing)
       b) Full-file XZ: one contiguous lzma.compress(entire file) in background
       Winner (smallest total) is emitted
    4. Decompression uses the main process (sequential by nature)
    """

    def __init__(self, table_path=None, trigrams_dir=None, verbose=True):
        """Initialize the compressor.

        Args:
            table_path: Path to a single trigram table (backward compat /
                        used for decompression of NC03 / TC01).
            trigrams_dir: Path to directory containing multiple .npz tables.
                          If None, defaults to trigrams/ next to this file.
            verbose: Print progress information.
        """
        self.verbose = verbose

        # Discover tables
        if trigrams_dir is None:
            trigrams_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "trigrams")

        self.table_entries = discover_trigram_tables(trigrams_dir)
        self.table_paths = [t[1] for t in self.table_entries]
        self.table_names = [t[0] for t in self.table_entries]

        # If no tables found in directory, fall back to single table
        if not self.table_entries and table_path:
            name = os.path.splitext(os.path.basename(table_path))[0]
            if name.startswith("trigram_"):
                name = name[8:]
            self.table_entries = [(name, os.path.abspath(table_path))]
            self.table_paths = [os.path.abspath(table_path)]
            self.table_names = [name]

        if not self.table_paths:
            raise ValueError(
                "No trigram tables found. Provide --table or put .npz files "
                "in trigrams/ directory.")

        # ---- Step 1: Load tables into shared memory ----
        self._shm_objects = []  # keep refs alive to prevent GC
        self._all_table_shm_info = []

        if self.verbose:
            print(f"Loading {len(self.table_paths)} tables into shared "
                  f"memory...", file=sys.stderr)

        for i, tp in enumerate(self.table_paths):
            if self.verbose:
                print(f"  [{i+1}/{len(self.table_paths)}] "
                      f"{self.table_names[i]} ({tp})", file=sys.stderr)
            shm_info, shm_objs = _load_table_to_shared_memory(
                tp, self.table_names[i], verbose=verbose)
            self._all_table_shm_info.append(shm_info)
            self._shm_objects.extend(shm_objs)

        # All tables must use the same tokenizer
        tokenizer_name = self._all_table_shm_info[0]["tokenizer_name"]
        for tinfo in self._all_table_shm_info[1:]:
            if tinfo["tokenizer_name"] != tokenizer_name:
                raise ValueError(
                    f"All tables must use same tokenizer. Got "
                    f"{tokenizer_name!r} and {tinfo['tokenizer_name']!r}")

        self._tokenizer_name = tokenizer_name

        # Load tokenizer in main process (for decompression)
        if self.verbose:
            print(f"Loading tokenizer: {tokenizer_name}", file=sys.stderr)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Build main-process decompression models from shared memory
        # (NO extra copy -- reuses the same shared memory segments)
        self.models = []
        self._main_shm_refs = []  # keep refs alive
        for tinfo in self._all_table_shm_info:
            table_dict = {
                "vocab_size": tinfo["vocab_size"],
                "tokenizer_name": tinfo["tokenizer_name"],
                "table_name": tinfo["table_name"],
            }
            for arr_name, arr_info in tinfo["arrays"].items():
                shm = shared_memory.SharedMemory(
                    name=arr_info["shm_name"], create=False)
                self._main_shm_refs.append(shm)
                arr = np.ndarray(
                    arr_info["shape"],
                    dtype=np.dtype(arr_info["dtype"]),
                    buffer=shm.buf,
                )
                table_dict[arr_name] = arr
            table_dict["uni_top_set"] = set(
                table_dict["uni_top_idx"].tolist())
            model = _WorkerAdaptiveModel(table_dict)
            self.models.append(model)

        # ---- Step 2: Create process pool ----
        num_tables = len(self.table_paths)
        self._num_workers = min(cpu_count(), num_tables + 1)

        if self.verbose:
            print(f"Creating ProcessPoolExecutor: {self._num_workers} workers "
                  f"(cpu_count={cpu_count()}, tables={num_tables})",
                  file=sys.stderr)

        self._pool = ProcessPoolExecutor(
            max_workers=self._num_workers,
            initializer=_worker_init,
            initargs=(self._all_table_shm_info, tokenizer_name),
        )

        # Warm up workers: ensure all have initialized before compression
        if self.verbose:
            print(f"Warming up {self._num_workers} worker processes...",
                  file=sys.stderr)
        warmup_futures = []
        for _ in range(self._num_workers):
            f = self._pool.submit(_worker_compress_with_lzma, b"warmup")
            warmup_futures.append(f)
        for f in warmup_futures:
            f.result()

        if self.verbose:
            print(f"Ready: {num_tables} tables | "
                  f"Names: {', '.join(self.table_names)} | "
                  f"Workers: {self._num_workers} processes",
                  file=sys.stderr)

        # Table name -> index mapping for NC05
        self._table_name_to_idx = {
            name: idx for idx, name in enumerate(self.table_names)
        }

    def shutdown(self):
        """Shutdown the process pool and clean up shared memory."""
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

        # Close main-process shared memory refs
        for shm in getattr(self, '_main_shm_refs', []):
            try:
                shm.close()
            except Exception:
                pass
        self._main_shm_refs = []

        # Close and unlink shared memory segments (owner refs)
        for shm in self._shm_objects:
            try:
                shm.close()
            except Exception:
                pass
            try:
                shm.unlink()
            except Exception:
                pass
        self._shm_objects = []

    # ---- per-chunk: test all trigram tables in TRUE PARALLEL ----

    def _compress_text_chunk_trigram_only(self, sub_data):
        """Test all trigram tables on a single text chunk (no lzma).

        Submits N trigram table tasks to the process pool simultaneously.
        The lzma comparison is handled by the accumulation logic in
        compress_bytes().

        Returns (table_idx, comp_data, winner_name) for the best trigram
        result, or (None, None, None) if all tables failed.
        """
        sub_len = len(sub_data)

        # Submit N trigram table tasks in parallel
        futures = []
        for ti in range(len(self.table_names)):
            future = self._pool.submit(
                _worker_compress_with_table, ti, sub_data)
            futures.append(future)

        # Wait for ALL results
        results = [f.result() for f in futures]

        # Find best trigram result
        best_tri_idx = None
        best_tri_data = None
        best_tri_size = sub_len + 1  # worse than raw

        for table_idx, method, comp_data in results:
            if method == METHOD_TRIGRAM and comp_data is not None:
                tri_size = len(comp_data)
                if tri_size < best_tri_size:
                    best_tri_idx = table_idx
                    best_tri_data = comp_data
                    best_tri_size = tri_size

        if best_tri_idx is not None:
            return (best_tri_idx, best_tri_data,
                    self.table_names[best_tri_idx])
        else:
            return (None, None, None)

    # ---- public API: compress text (TC01, backward compat) ----

    def compress(self, text):
        """Compress a text string -> TC01 format bytes (uses first table)."""
        if not text:
            return MAGIC_TEXT + struct.pack('>II', 0, 0)
        chunk_bytes = text.encode('latin-1')
        num_tokens, stream = _trigram_compress_chunk(
            self.models[0], self.tokenizer, chunk_bytes)
        bit_count = len(stream) * 8
        return MAGIC_TEXT + struct.pack('>II', num_tokens, bit_count) + stream

    def decompress_text(self, data):
        """Decompress TC01 format -> text string."""
        if len(data) < 12:
            raise ValueError("Data too short")
        magic = data[:4]
        if magic != MAGIC_TEXT:
            raise ValueError(f"Expected TC01, got {magic!r}")
        num_tokens, _ = struct.unpack('>II', data[4:12])
        if num_tokens == 0:
            return ""
        raw = _trigram_decompress_chunk(
            self.models[0], self.tokenizer, data[12:], num_tokens)
        return raw.decode('latin-1')

    # ---- public API: compress bytes (NC05 multi-table) ----

    def compress_bytes(self, data: bytes) -> bytes:
        """Compress raw bytes -> NC05 multi-table format.

        1. Segment into binary vs text regions
        2. Binary regions -> lzma (always)
        3. Two competing plans built simultaneously:
           a) Individual: best of trigram/lzma per text chunk, lzma per binary
           b) Full-file XZ: one contiguous lzma of entire input
           Emits whichever plan produces smaller total output

        NC05 format:
          [4B] Magic "NC05"
          [4B] Original total size (uint32 BE)
          [2B] Number of tables (uint16 BE)
          Per table:
            [2B] Name length (uint16 BE)
            [NB] Table name (UTF-8)
          [4B] Number of entries (uint32 BE)
          Per entry:
            [1B] Method: 'B' binary/lzma, 'T' trigram, 'L' text/lzma
            [1B] Table index (only meaningful for 'T', 0 otherwise)
            [4B] Original size (uint32 BE)
            [4B] Compressed size (uint32 BE)
            For 'B'/'L': raw compressed data
            For 'T': [4B] token_count (uint32 BE) + stream
        """
        total_size = len(data)
        if total_size == 0:
            return MAGIC_CHUNK + struct.pack('>II', 0, 0)

        # Step 1: segment binary vs text
        segments = _segment_chunks(data)

        total_binary = sum(l for t, _, l in segments if t == CHUNK_TYPE_BINARY)
        total_text = sum(l for t, _, l in segments if t == CHUNK_TYPE_TEXT)
        n_bin_segs = sum(1 for t, _, _ in segments if t == CHUNK_TYPE_BINARY)
        n_txt_segs = sum(1 for t, _, _ in segments if t == CHUNK_TYPE_TEXT)

        if self.verbose:
            print(f"Segments: {len(segments)} ({n_bin_segs} binary: "
                  f"{total_binary} bytes, {n_txt_segs} text: "
                  f"{total_text} bytes)", file=sys.stderr)
            print(f"Tables: {len(self.table_names)} "
                  f"({', '.join(self.table_names)}) | "
                  f"Workers: {self._num_workers} processes",
                  file=sys.stderr)

        # Step 2: build individual per-chunk entries AND full-file XZ
        # Strategy: compute best per-chunk compression (trigram vs lzma)
        # for each chunk. Simultaneously, compute one contiguous lzma of
        # the entire file. At the end, emit whichever is smaller.

        # Submit full-file XZ to worker pool (runs in background while
        # we process individual chunks)
        full_xz_future = self._pool.submit(lzma.compress, data)

        individual_entries = []  # (method, table_idx, orig_size, comp_data)
        individual_total_comp = 0
        bytes_done = 0
        trigram_wins = 0
        lzma_text_wins = 0
        lzma_bin_wins = 0
        table_win_counts = {name: 0 for name in self.table_names}

        for seg_type, offset, length in segments:
            seg_data = data[offset:offset + length]

            if seg_type == CHUNK_TYPE_BINARY:
                # Binary -> lzma for individual plan
                comp = lzma.compress(seg_data)
                individual_entries.append((METHOD_BINARY, 0, length, comp))
                individual_total_comp += len(comp)
                lzma_bin_wins += 1

                if self.verbose:
                    ratio = len(comp) / length if length > 0 else 0
                    overall = 100 * bytes_done / total_size
                    print(f"  Binary: {length} -> {len(comp)} ({ratio:.1%})"
                          f"  [total: {overall:.1f}%]", file=sys.stderr)
                bytes_done += length

            else:
                # Text -> dynamic chunk sizing, best of trigram/lzma per chunk
                chunk_size = max(2048, min(65536, length // 10))

                for sub_off in range(0, length, chunk_size):
                    sub_end = min(sub_off + chunk_size, length)
                    sub_data = seg_data[sub_off:sub_end]
                    sub_len = len(sub_data)

                    overall = 100 * bytes_done / total_size

                    # Test all trigram tables in parallel (workers)
                    tri_idx, tri_data, tri_name = \
                        self._compress_text_chunk_trigram_only(sub_data)

                    # Per-chunk lzma (main process)
                    chunk_lzma = lzma.compress(sub_data)

                    # Pick best individual compression for this chunk
                    if tri_data is not None and len(tri_data) <= len(chunk_lzma):
                        individual_entries.append(
                            (METHOD_TRIGRAM, tri_idx, sub_len, tri_data))
                        individual_total_comp += len(tri_data)
                        trigram_wins += 1
                        table_win_counts[tri_name] = \
                            table_win_counts.get(tri_name, 0) + 1
                        tag = f"T:{tri_name}"
                        comp_size = len(tri_data)
                    else:
                        individual_entries.append(
                            (METHOD_LZMA, 0, sub_len, chunk_lzma))
                        individual_total_comp += len(chunk_lzma)
                        lzma_text_wins += 1
                        tag = "L"
                        comp_size = len(chunk_lzma)

                    if self.verbose:
                        ratio = comp_size / sub_len if sub_len > 0 else 0
                        print(f"  Text: {sub_len} -> {comp_size} "
                              f"({tag}, {ratio:.1%})"
                              f"  [total: {overall:.1f}%]",
                              file=sys.stderr)

                    bytes_done += sub_len

        # Collect full-file contiguous XZ result
        full_xz = full_xz_future.result()

        if self.verbose:
            full_ratio = len(full_xz) / total_size if total_size else 0
            ind_ratio = individual_total_comp / total_size if total_size else 0
            print(f"  Full-file XZ: {total_size} -> {len(full_xz)} "
                  f"({full_ratio:.1%})", file=sys.stderr)
            print(f"  Individual entries: {total_size} -> "
                  f"{individual_total_comp} ({ind_ratio:.1%})",
                  file=sys.stderr)

        # Final decision: full-file XZ vs individual entries
        if len(full_xz) <= individual_total_comp:
            entries = [(METHOD_LZMA, 0, total_size, full_xz)]
            if self.verbose:
                saved = individual_total_comp - len(full_xz)
                print(f"  Winner: full-file XZ (saves {saved} bytes)",
                      file=sys.stderr)
        else:
            entries = individual_entries
            if self.verbose:
                saved = len(full_xz) - individual_total_comp
                print(f"  Winner: individual entries (saves {saved} bytes)",
                      file=sys.stderr)
                print(f"  Breakdown: {lzma_bin_wins} binary(lzma), "
                      f"{trigram_wins} text(trigram), "
                      f"{lzma_text_wins} text(lzma)", file=sys.stderr)
                if trigram_wins > 0:
                    wins_str = ", ".join(
                        f"{name}={cnt}"
                        for name, cnt in table_win_counts.items()
                        if cnt > 0)
                    print(f"  Table wins: {wins_str}", file=sys.stderr)

        # Assemble NC05
        num_entries = len(entries)

        # Header
        header_parts = [MAGIC_CHUNK, struct.pack('>I', total_size)]

        # Table directory
        n_tables = len(self.table_names)
        header_parts.append(struct.pack('>H', n_tables))
        for name in self.table_names:
            name_bytes = name.encode('utf-8')
            header_parts.append(struct.pack('>H', len(name_bytes)))
            header_parts.append(name_bytes)

        # Entry count
        header_parts.append(struct.pack('>I', num_entries))

        # Entries
        for method, table_idx, orig_size, comp_data in entries:
            header_parts.append(struct.pack('>BBII', method, table_idx,
                                            orig_size, len(comp_data)))
            header_parts.append(comp_data)

        return b''.join(header_parts)

    # ---- decompression: NC05 ----

    def _decompress_nc05(self, data: bytes) -> bytes:
        """Decompress NC05 multi-table format -> raw bytes."""
        if len(data) < 10:
            raise ValueError("NC05 data too short")

        pos = 4  # skip magic
        total_size = struct.unpack('>I', data[pos:pos + 4])[0]
        pos += 4

        if total_size == 0:
            return b""

        # Read table directory
        n_tables = struct.unpack('>H', data[pos:pos + 2])[0]
        pos += 2
        file_table_names = []
        for _ in range(n_tables):
            name_len = struct.unpack('>H', data[pos:pos + 2])[0]
            pos += 2
            name = data[pos:pos + name_len].decode('utf-8')
            pos += name_len
            file_table_names.append(name)

        # Map file table indices to our loaded model indices
        table_map = {}
        for fi, fname in enumerate(file_table_names):
            if fname in self._table_name_to_idx:
                table_map[fi] = self._table_name_to_idx[fname]
            else:
                raise ValueError(
                    f"Compressed file requires table '{fname}' which is not "
                    f"loaded. Available: {', '.join(self.table_names)}")

        # Read entries
        num_entries = struct.unpack('>I', data[pos:pos + 4])[0]
        pos += 4

        output_parts = []
        bytes_done = 0

        for ci in range(num_entries):
            method, file_table_idx, orig_size, comp_size = struct.unpack(
                '>BBII', data[pos:pos + 10])
            pos += 10
            comp_data = data[pos:pos + comp_size]
            pos += comp_size

            if method == METHOD_BINARY:
                mname = "B"
            elif method == METHOD_TRIGRAM:
                tname = file_table_names[file_table_idx]
                mname = f"T:{tname}"
            else:
                mname = "L"

            if self.verbose:
                overall = 100 * bytes_done / total_size if total_size else 0
                print(f"\r  Chunk {ci+1}/{num_entries}: {comp_size} -> "
                      f"{orig_size} ({mname})  [total: {overall:.1f}%]",
                      end="", file=sys.stderr)

            if method == METHOD_BINARY or method == METHOD_LZMA:
                output_parts.append(lzma.decompress(comp_data))
            elif method == METHOD_TRIGRAM:
                model_idx = table_map[file_table_idx]
                num_tokens = struct.unpack('>I', comp_data[:4])[0]
                stream = comp_data[4:]
                output_parts.append(
                    _trigram_decompress_chunk(
                        self.models[model_idx], self.tokenizer,
                        stream, num_tokens))
            else:
                raise ValueError(f"Unknown method: {method:#x}")

            bytes_done += orig_size

        if self.verbose:
            print(f"\r  Done: {num_entries} chunks, {total_size} bytes"
                  f"  [total: 100.0%]       ", file=sys.stderr)
            print(file=sys.stderr)

        return b''.join(output_parts)

    # ---- decompression: NC03 (backward compat) ----

    def _decompress_nc03(self, data: bytes) -> bytes:
        """Decompress NC03 format -> raw bytes (backward compat)."""
        if len(data) < 12:
            raise ValueError("NC03 data too short")

        total_size, num_entries = struct.unpack('>II', data[4:12])
        if num_entries == 0:
            return b""

        pos = 12
        output_parts = []
        bytes_done = 0

        for ci in range(num_entries):
            method, orig_size, comp_size = struct.unpack(
                '>BII', data[pos:pos + 9])
            pos += 9
            comp_data = data[pos:pos + comp_size]
            pos += comp_size

            if method == METHOD_BINARY:
                mname = "B"
            elif method == METHOD_TRIGRAM:
                mname = "T"
            else:
                mname = "L"

            if self.verbose:
                overall = 100 * bytes_done / total_size if total_size else 0
                print(f"\r  Chunk {ci+1}/{num_entries}: {comp_size} -> "
                      f"{orig_size} ({mname})  [total: {overall:.1f}%]",
                      end="", file=sys.stderr)

            if method == METHOD_BINARY or method == METHOD_LZMA:
                output_parts.append(lzma.decompress(comp_data))
            elif method == METHOD_TRIGRAM:
                num_tokens = struct.unpack('>I', comp_data[:4])[0]
                stream = comp_data[4:]
                # Use first model (NC03 only had one table)
                output_parts.append(
                    _trigram_decompress_chunk(
                        self.models[0], self.tokenizer,
                        stream, num_tokens))
            else:
                raise ValueError(f"Unknown method: {method:#x}")

            bytes_done += orig_size

        if self.verbose:
            print(f"\r  Done: {num_entries} chunks, {total_size} bytes"
                  f"  [total: 100.0%]       ", file=sys.stderr)
            print(file=sys.stderr)

        return b''.join(output_parts)

    # ---- decompression: auto-detect ----

    def decompress_bytes(self, data: bytes) -> bytes:
        """Decompress NC03 or NC05 format -> raw bytes."""
        magic = data[:4]
        if magic == MAGIC_CHUNK:
            return self._decompress_nc05(data)
        elif magic == MAGIC_NC03:
            return self._decompress_nc03(data)
        else:
            raise ValueError(f"Expected NC03 or NC05, got {magic!r}")

    # ---- unified API ----

    def decompress(self, data: bytes):
        """Auto-detect format and decompress."""
        magic = data[:4]
        if magic == MAGIC_TEXT:
            return self.decompress_text(data)
        elif magic == MAGIC_CHUNK:
            return self._decompress_nc05(data)
        elif magic == MAGIC_NC03:
            return self._decompress_nc03(data)
        else:
            raise ValueError(f"Unknown format magic: {magic!r}")
