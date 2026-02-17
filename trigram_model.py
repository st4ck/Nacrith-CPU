"""
Sparse trigram model with adaptive bigram+trigram counting.

Instead of building full 49K probability arrays, we work with sparse
representations: only top-K tokens per context + one "rest" bucket.
The CDF has ~TOP_K+1 entries instead of 49,153.

The arithmetic coder encodes a "symbol index" in range [0, TOP_K].
Index 0..TOP_K-1 = specific tokens, index TOP_K = rest bucket.
We maintain a mapping from token_id → symbol_index per step.
"""

import sys
from collections import Counter, defaultdict

import numpy as np

CDF_TOTAL = 1 << 16
MIN_PROB = 1

LAMBDA_TRI = 0.70
LAMBDA_BI = 0.25
LAMBDA_UNI = 0.05
LAMBDA_BI_ONLY = 0.80
LAMBDA_UNI_ONLY = 0.20

MAX_ADAPTIVE_WEIGHT = 0.35
ADAPTIVE_RAMP_TOKENS = 800

# How many tokens to keep in the sparse CDF (rest go into one bucket)
SPARSE_TOP_K = 512


class TrigramModel:
    def __init__(self, table_path: str, verbose: bool = True):
        self.verbose = verbose
        self._load_table(table_path)

    def _load_table(self, path: str):
        if self.verbose:
            print(f"Loading trigram table: {path}", file=sys.stderr)
        data = np.load(path, allow_pickle=True)
        self.vocab_size = int(data["vocab_size"][0])
        self.tokenizer_name = str(data["tokenizer_name"][0])
        self.unigram_probs = data["unigram_probs"].astype(np.float64)
        self.bi_ctx_keys = data["bigram_context_keys"]
        self.bi_top_tokens = data["bigram_top_tokens"]
        self.bi_top_probs = data["bigram_top_probs"].astype(np.float64)
        self.bi_remaining = data["bigram_remaining_mass"].astype(np.float64)
        self.tri_ctx_keys = data["trigram_context_keys"]
        self.tri_top_tokens = data["trigram_top_tokens"]
        self.tri_top_probs = data["trigram_top_probs"].astype(np.float64)
        self.tri_remaining = data["trigram_remaining_mass"].astype(np.float64)

        # Precompute sorted unigram top-K indices for fast sparse building
        self._uni_top_idx = np.argsort(self.unigram_probs)[::-1][:SPARSE_TOP_K].copy()
        self._uni_top_set = set(self._uni_top_idx.tolist())

        if self.verbose:
            print(f"  Vocab size: {self.vocab_size:,}", file=sys.stderr)
            print(f"  Bigram contexts: {len(self.bi_ctx_keys):,}", file=sys.stderr)
            print(f"  Trigram contexts: {len(self.tri_ctx_keys):,}", file=sys.stderr)
            print(f"  Sparse top-K: {SPARSE_TOP_K}", file=sys.stderr)

    def _lookup_bigram(self, prev1):
        idx = np.searchsorted(self.bi_ctx_keys, prev1)
        if idx < len(self.bi_ctx_keys) and self.bi_ctx_keys[idx] == prev1:
            return self.bi_top_tokens[idx], self.bi_top_probs[idx], self.bi_remaining[idx]
        return None

    def _lookup_trigram(self, prev2, prev1):
        key = np.uint32((prev2 & 0xFFFF) << 16 | (prev1 & 0xFFFF))
        idx = np.searchsorted(self.tri_ctx_keys, key)
        if idx < len(self.tri_ctx_keys) and self.tri_ctx_keys[idx] == key:
            return self.tri_top_tokens[idx], self.tri_top_probs[idx], self.tri_remaining[idx]
        return None


class AdaptiveTrigramModel:
    def __init__(self, table_path: str, verbose: bool = True):
        self.static = TrigramModel(table_path, verbose=verbose)
        self.vocab_size = self.static.vocab_size
        self.tokenizer_name = self.static.tokenizer_name
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

    def get_sparse_cdf(self, context):
        """Return (token_ids, cdf) where token_ids has K entries and cdf has K+1.

        token_ids[i] is the actual token id for symbol index i.
        Symbol index K (= len(token_ids)) is the "rest" bucket covering all
        tokens NOT in token_ids.

        cdf[0] = 0, cdf[-1] = CDF_TOTAL.
        """
        s = self.static

        # --- Step 1: Collect candidate token set and their raw probabilities ---
        # Start with unigram top tokens
        candidates = set(s._uni_top_set)

        # Gather raw component probabilities for candidates
        # We'll store: for each candidate token, its static prob
        # Also collect all tokens from bigram/trigram lookups + adaptive

        bi_result = None
        tri_result = None

        if len(context) >= 1:
            prev1 = context[-1]
            bi_result = s._lookup_bigram(prev1)
            if bi_result is not None:
                valid = bi_result[1] > 0
                candidates.update(bi_result[0][valid].tolist())

            if len(context) >= 2:
                prev2 = context[-2]
                tri_result = s._lookup_trigram(prev2, prev1)
                if tri_result is not None:
                    valid = tri_result[1] > 0
                    candidates.update(tri_result[0][valid].tolist())

        # Add adaptive tokens
        lambda_a = min(MAX_ADAPTIVE_WEIGHT, self.tokens_seen / ADAPTIVE_RAMP_TOKENS)
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

        # Convert to sorted array for deterministic order
        token_ids = np.array(sorted(candidates), dtype=np.int64)
        n = len(token_ids)

        # --- Step 2: Build probability for each candidate ---
        # Unigram probs for candidates
        uni_probs = s.unigram_probs[token_ids]

        # Static interpolation
        if len(context) >= 2 and tri_result is not None and bi_result is not None:
            # Full trigram + bigram + unigram
            bi_probs = uni_probs * bi_result[2]
            # Override with bigram top entries that appear in our candidates
            bi_tok = bi_result[0]
            bi_p = bi_result[1]
            bi_valid = bi_p > 0
            if bi_valid.any():
                _map_into(bi_probs, token_ids, bi_tok[bi_valid], bi_p[bi_valid])

            tri_probs = bi_probs * tri_result[2]
            tri_tok = tri_result[0]
            tri_p = tri_result[1]
            tri_valid = tri_p > 0
            if tri_valid.any():
                _map_into(tri_probs, token_ids, tri_tok[tri_valid], tri_p[tri_valid])

            static_probs = LAMBDA_TRI * tri_probs + LAMBDA_BI * bi_probs + LAMBDA_UNI * uni_probs

        elif len(context) >= 2 and tri_result is not None:
            tri_probs = uni_probs * tri_result[2]
            tri_tok = tri_result[0]
            tri_p = tri_result[1]
            tri_valid = tri_p > 0
            if tri_valid.any():
                _map_into(tri_probs, token_ids, tri_tok[tri_valid], tri_p[tri_valid])
            static_probs = (LAMBDA_TRI + LAMBDA_BI) * tri_probs + LAMBDA_UNI * uni_probs

        elif bi_result is not None:
            bi_probs = uni_probs * bi_result[2]
            bi_tok = bi_result[0]
            bi_p = bi_result[1]
            bi_valid = bi_p > 0
            if bi_valid.any():
                _map_into(bi_probs, token_ids, bi_tok[bi_valid], bi_p[bi_valid])
            static_probs = LAMBDA_BI_ONLY * bi_probs + LAMBDA_UNI_ONLY * uni_probs

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

            final_probs = (1.0 - lambda_a) * static_probs + lambda_a * adaptive
        else:
            final_probs = static_probs

        # --- Step 4: Compute rest mass (probability not in our candidates) ---
        final_probs = np.maximum(final_probs, 1e-10)
        candidate_sum = final_probs.sum()
        rest_mass = max(1e-10, 1.0 - candidate_sum)

        # --- Step 5: Build integer CDF with rest bucket at end ---
        # We have n candidates + 1 rest bucket = n+1 symbols
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
        """Map token_id to its rank among non-excluded tokens.

        rank = how many tokens with id < token_id are NOT in excluded_sorted.
        excluded_sorted must be a sorted list.
        """
        rest_size = self.vocab_size - len(excluded_sorted)
        if rest_size <= 0:
            rest_size = 1

        # rank = token_id minus (number of excluded tokens < token_id)
        # excluded_sorted is sorted, binary search for count
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
    # Both are sorted, use merge-style
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

    # Map observed tokens into our sparse set
    idx = np.searchsorted(token_ids, obs_tokens)
    valid = (idx < n) & (token_ids[idx] == obs_tokens)
    dist[idx[valid]] = smoothed[valid]

    s = dist.sum()
    if s > 0:
        dist /= s
    return dist
