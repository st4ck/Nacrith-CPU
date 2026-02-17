#!/usr/bin/env python3
"""
Nacrith v7.6 -- TRUE PARALLEL multiprocessing multi-table compressor.

Auto-loads ALL trigram tables from trigrams/ directory into SHARED MEMORY,
then tests each table against every text sub-chunk in TRUE PARALLEL using
ProcessPoolExecutor (multiple CPU cores, bypasses GIL).

Architecture:
  - Tables loaded into shared memory ONCE (zero-copy for workers)
  - ProcessPoolExecutor with N workers (N = min(cpu_count, tables+1))
  - Per chunk: N trigram + 1 lzma tasks submitted SIMULTANEOUSLY
  - All CPU cores utilized for CPU-bound arithmetic coding

Text regions  -> parallel test of N trigram tables + lzma (best wins)
Binary regions -> lzma

Usage:
    nacrith.py c INPUT OUTPUT [--table TABLE] [--trigrams-dir DIR]
    nacrith.py d INPUT OUTPUT [--table TABLE] [--trigrams-dir DIR]
    nacrith.py benchmark INPUT [--table TABLE] [--trigrams-dir DIR]
    nacrith.py build-table --corpus wikitext-103 --output table.npz

Requirements:
    pip install numpy transformers
    python setup.py build_ext --inplace   (builds C arithmetic coder)
"""

import argparse
import gzip
import lzma
import os
import sys
import time

from compressor import TrigramCompressor

DEFAULT_TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "trigram_table.npz")
DEFAULT_TRIGRAMS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "trigrams")


def fmt(num_bytes):
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    else:
        return f"{num_bytes / (1024 * 1024):.2f} MB"


def cmd_compress(args):
    trigrams_dir = args.trigrams_dir if args.trigrams_dir else DEFAULT_TRIGRAMS_DIR
    comp = TrigramCompressor(
        table_path=args.table,
        trigrams_dir=trigrams_dir,
        verbose=not args.quiet)

    with open(args.input, "rb") as f:
        data = f.read()

    original_size = len(data)
    if not args.quiet:
        print(f"Input:  {args.input} ({fmt(original_size)})", file=sys.stderr)

    t0 = time.time()
    compressed = comp.compress_bytes(data)
    elapsed = time.time() - t0

    with open(args.output, "wb") as f:
        f.write(compressed)

    comp_size = len(compressed)
    ratio = comp_size / original_size if original_size > 0 else 0
    bpb = (comp_size * 8) / original_size if original_size > 0 else 0

    if not args.quiet:
        print(f"Output: {args.output} ({fmt(comp_size)})", file=sys.stderr)
        print(f"Ratio:  {ratio:.4f} ({100*ratio:.1f}%)  "
              f"{bpb:.4f} bits/byte  {elapsed:.1f}s", file=sys.stderr)

    comp.shutdown()


def cmd_decompress(args):
    trigrams_dir = args.trigrams_dir if args.trigrams_dir else DEFAULT_TRIGRAMS_DIR
    comp = TrigramCompressor(
        table_path=args.table,
        trigrams_dir=trigrams_dir,
        verbose=not args.quiet)

    with open(args.input, "rb") as f:
        data = f.read()

    if not args.quiet:
        print(f"Input:  {args.input} ({fmt(len(data))})", file=sys.stderr)

    t0 = time.time()
    result = comp.decompress(data)
    elapsed = time.time() - t0

    if isinstance(result, str):
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
        out_size = len(result.encode("utf-8"))
    else:
        with open(args.output, "wb") as f:
            f.write(result)
        out_size = len(result)

    if not args.quiet:
        print(f"Output: {args.output} ({fmt(out_size)})", file=sys.stderr)
        print(f"Time:   {elapsed:.1f}s", file=sys.stderr)

    comp.shutdown()


def cmd_benchmark(args):
    trigrams_dir = args.trigrams_dir if args.trigrams_dir else DEFAULT_TRIGRAMS_DIR
    comp = TrigramCompressor(
        table_path=args.table,
        trigrams_dir=trigrams_dir,
        verbose=False)

    with open(args.input, "rb") as f:
        data = f.read()

    original_size = len(data)
    print(f"File: {args.input} ({fmt(original_size)})")
    print(f"{'='*70}")

    results = []

    # gzip
    t0 = time.time()
    gz = gzip.compress(data, compresslevel=9)
    results.append(("gzip -9", len(gz), time.time() - t0))

    # lzma
    t0 = time.time()
    xz = lzma.compress(data, preset=9)
    results.append(("lzma -9", len(xz), time.time() - t0))

    # nacrith (parallel multi-table)
    t0 = time.time()
    nc = comp.compress_bytes(data)
    nc_time = time.time() - t0
    results.append(("nacrith", len(nc), nc_time))

    # roundtrip
    restored = comp.decompress(nc)
    lossless = (restored == data)

    print(f"\n{'Method':<12} {'Size':>10} {'Ratio':>8} {'bits/B':>8} {'Time':>8}")
    print(f"{'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
    print(f"{'original':<12} {fmt(original_size):>10}")

    for name, size, elapsed in results:
        ratio = size / original_size
        bpb = (size * 8) / original_size
        print(f"{name:<12} {fmt(size):>10} {ratio:>7.1%} {bpb:>7.4f} {elapsed:>7.1f}s")

    print(f"\nLossless: {'PASS' if lossless else 'FAIL'}")
    if not lossless:
        if isinstance(restored, bytes) and isinstance(data, bytes):
            if len(data) != len(restored):
                print(f"  Length: {len(data)} vs {len(restored)}")
            else:
                for i in range(min(len(data), len(restored))):
                    if data[i] != restored[i]:
                        print(f"  First diff at byte {i}")
                        break

    comp.shutdown()


def cmd_build_table(args):
    from build_table import main as build_main
    # Forward args to build_table
    build_args = []
    if args.corpus:
        build_args += ["--corpus", args.corpus]
    if args.file:
        build_args += ["--file", args.file]
    build_args += ["--output", args.output]
    sys.argv = ["build_table"] + build_args
    build_main()


def main():
    parser = argparse.ArgumentParser(
        prog="nacrith",
        description="Nacrith v7.6 -- TRUE PARALLEL multiprocessing compressor")
    parser.add_argument("--table", type=str, default=DEFAULT_TABLE,
                        help="Single trigram table path (fallback if no "
                             "trigrams/ directory)")
    parser.add_argument("--trigrams-dir", type=str, default=None,
                        help="Directory containing .npz trigram tables "
                             "(default: trigrams/ next to this script)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress progress output")

    sub = parser.add_subparsers(dest="command")

    # c / compress
    p_c = sub.add_parser("c", aliases=["compress"], help="Compress a file")
    p_c.add_argument("input", help="Input file")
    p_c.add_argument("output", help="Output compressed file")

    # d / decompress
    p_d = sub.add_parser("d", aliases=["decompress"],
                         help="Decompress a file")
    p_d.add_argument("input", help="Input compressed file")
    p_d.add_argument("output", help="Output restored file")

    # benchmark
    p_b = sub.add_parser("benchmark", aliases=["b"],
                         help="Compare against gzip/lzma")
    p_b.add_argument("input", help="Input file")

    # build-table
    p_bt = sub.add_parser("build-table",
                          help="Build trigram table from corpus")
    p_bt.add_argument("--corpus", choices=["wikitext-2", "wikitext-103"],
                      help="Standard corpus to download")
    p_bt.add_argument("--file", type=str, help="Local text file as corpus")
    p_bt.add_argument("--output", type=str, default="trigram_table.npz",
                      help="Output .npz path")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command in ("c", "compress"):
        cmd_compress(args)
    elif args.command in ("d", "decompress"):
        cmd_decompress(args)
    elif args.command in ("benchmark", "b"):
        cmd_benchmark(args)
    elif args.command == "build-table":
        cmd_build_table(args)


if __name__ == "__main__":
    main()
