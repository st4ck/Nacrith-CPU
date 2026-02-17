#!/usr/bin/env python3
"""
Benchmark text files one by one: gzip vs lzma vs Nacrith v7.6
"""
import glob
import gzip
import lzma
import os
import sys
import time
import json

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compressor import TrigramCompressor


def format_size(bytes_val):
    """Format bytes as human-readable."""
    if bytes_val < 1024:
        return f"{bytes_val} B"
    elif bytes_val < 1024 * 1024:
        return f"{bytes_val / 1024:.1f} KB"
    else:
        return f"{bytes_val / (1024 * 1024):.1f} MB"


def format_ratio(comp, orig):
    """Format compression ratio as percentage."""
    if orig == 0:
        return "0.0%"
    return f"{100 * comp / orig:.2f}%"


def format_bpb(comp, orig):
    """Format bits per byte."""
    if orig == 0:
        return "0.00"
    return f"{8 * comp / orig:.2f}"


def benchmark_file(file_path, compressor):
    """Benchmark a single file with all three compressors."""
    filename = os.path.basename(file_path)

    # Read file
    with open(file_path, 'rb') as f:
        data = f.read()

    orig_size = len(data)

    print(f"\n{'='*70}")
    print(f"File: {filename}")
    print(f"Original size: {format_size(orig_size)}")
    print(f"{'='*70}")

    results = {
        'filename': filename,
        'original_size': orig_size,
        'compressors': {}
    }

    # 1. gzip -9
    print(f"\n[1/3] Testing gzip -9...")
    t0 = time.time()
    gzip_data = gzip.compress(data, compresslevel=9)
    gzip_time = time.time() - t0
    gzip_size = len(gzip_data)

    # Verify lossless
    assert gzip.decompress(gzip_data) == data, "gzip decompression failed"

    print(f"  Compressed: {format_size(gzip_size)} ({format_ratio(gzip_size, orig_size)})")
    print(f"  Bits/byte: {format_bpb(gzip_size, orig_size)}")
    print(f"  Time: {gzip_time:.2f}s")

    results['compressors']['gzip'] = {
        'size': gzip_size,
        'ratio': gzip_size / orig_size,
        'bpb': 8 * gzip_size / orig_size,
        'time': gzip_time
    }

    # 2. lzma (xz)
    print(f"\n[2/3] Testing lzma (xz)...")
    t0 = time.time()
    lzma_data = lzma.compress(data, preset=9)
    lzma_time = time.time() - t0
    lzma_size = len(lzma_data)

    # Verify lossless
    assert lzma.decompress(lzma_data) == data, "lzma decompression failed"

    print(f"  Compressed: {format_size(lzma_size)} ({format_ratio(lzma_size, orig_size)})")
    print(f"  Bits/byte: {format_bpb(lzma_size, orig_size)}")
    print(f"  Time: {lzma_time:.2f}s")

    results['compressors']['lzma'] = {
        'size': lzma_size,
        'ratio': lzma_size / orig_size,
        'bpb': 8 * lzma_size / orig_size,
        'time': lzma_time
    }

    # 3. Nacrith v7.6
    print(f"\n[3/3] Testing Nacrith v7.6...")
    t0 = time.time()
    nacrith_data = compressor.compress_bytes(data)
    nacrith_time = time.time() - t0
    nacrith_size = len(nacrith_data)

    print(f"  Compressed: {format_size(nacrith_size)} ({format_ratio(nacrith_size, orig_size)})")
    print(f"  Bits/byte: {format_bpb(nacrith_size, orig_size)}")
    print(f"  Time: {nacrith_time:.2f}s")

    # Verify lossless
    print(f"  Verifying lossless decompression...")
    decompressed = compressor.decompress_bytes(nacrith_data)
    if decompressed == data:
        print(f"  ✓ Lossless: PASS")
    else:
        print(f"  ✗ Lossless: FAIL")
        results['compressors']['nacrith'] = {
            'size': nacrith_size,
            'ratio': nacrith_size / orig_size,
            'bpb': 8 * nacrith_size / orig_size,
            'time': nacrith_time,
            'lossless': False
        }
        return results

    results['compressors']['nacrith'] = {
        'size': nacrith_size,
        'ratio': nacrith_size / orig_size,
        'bpb': 8 * nacrith_size / orig_size,
        'time': nacrith_time,
        'lossless': True
    }

    # Summary comparison
    print(f"\n{'─'*70}")
    print(f"SUMMARY:")
    print(f"  gzip:    {format_size(gzip_size):>10} ({format_ratio(gzip_size, orig_size):>7}, {format_bpb(gzip_size, orig_size):>5} bpb)")
    print(f"  lzma:    {format_size(lzma_size):>10} ({format_ratio(lzma_size, orig_size):>7}, {format_bpb(lzma_size, orig_size):>5} bpb)")
    print(f"  Nacrith: {format_size(nacrith_size):>10} ({format_ratio(nacrith_size, orig_size):>7}, {format_bpb(nacrith_size, orig_size):>5} bpb)")

    # Best compressor
    best = min([('gzip', gzip_size), ('lzma', lzma_size), ('Nacrith', nacrith_size)],
               key=lambda x: x[1])
    print(f"\n  Winner: {best[0]} with {format_size(best[1])}")

    # Nacrith comparison
    if nacrith_size <= lzma_size:
        improvement = 100 * (lzma_size - nacrith_size) / lzma_size
        print(f"  Nacrith beats lzma by {improvement:.1f}%")
    else:
        worse = 100 * (nacrith_size - lzma_size) / lzma_size
        print(f"  Nacrith is {worse:.1f}% larger than lzma")

    return results


def main():
    benchmark_dir = "benchmark_files"

    if not os.path.isdir(benchmark_dir):
        print(f"Error: directory '{benchmark_dir}' not found")
        return 1

    # Find all .txt files
    txt_files = sorted(glob.glob(os.path.join(benchmark_dir, "*.txt")))

    if not txt_files:
        print(f"Error: no .txt files found in '{benchmark_dir}'")
        return 1

    print(f"Found {len(txt_files)} text files to benchmark")

    # Initialize compressor
    print(f"\nInitializing Nacrith compressor...")
    global compressor
    compressor = TrigramCompressor(verbose=False)

    # Benchmark each file
    all_results = []
    for i, file_path in enumerate(txt_files, 1):
        print(f"\n\n{'#'*70}")
        print(f"# [{i}/{len(txt_files)}] Processing: {os.path.basename(file_path)}")
        print(f"{'#'*70}")

        results = benchmark_file(file_path, compressor)
        all_results.append(results)

    # Save results
    output_file = "benchmark_text_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\n{'='*70}")
    print(f"ALL BENCHMARKS COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_file}")

    # Print summary table
    print(f"\n\nSUMMARY TABLE:")
    print(f"{'─'*70}")
    print(f"{'File':<30} {'Orig':>8} {'gzip':>8} {'lzma':>8} {'Nacrith':>8} {'Winner':>10}")
    print(f"{'─'*70}")

    for r in all_results:
        orig = r['original_size']
        gzip_size = r['compressors']['gzip']['size']
        lzma_size = r['compressors']['lzma']['size']
        nacrith_size = r['compressors']['nacrith']['size']

        best = min([('gzip', gzip_size), ('lzma', lzma_size), ('Nacrith', nacrith_size)],
                   key=lambda x: x[1])

        filename = r['filename']
        if len(filename) > 29:
            filename = filename[:26] + "..."

        print(f"{filename:<30} {format_size(orig):>8} "
              f"{format_size(gzip_size):>8} {format_size(lzma_size):>8} "
              f"{format_size(nacrith_size):>8} {best[0]:>10}")

    print(f"{'─'*70}")

    # Count wins
    gzip_wins = sum(1 for r in all_results
                    if r['compressors']['gzip']['size'] <= min(
                        r['compressors']['lzma']['size'],
                        r['compressors']['nacrith']['size']))
    lzma_wins = sum(1 for r in all_results
                    if r['compressors']['lzma']['size'] <= min(
                        r['compressors']['gzip']['size'],
                        r['compressors']['nacrith']['size']))
    nacrith_wins = sum(1 for r in all_results
                       if r['compressors']['nacrith']['size'] <= min(
                           r['compressors']['gzip']['size'],
                           r['compressors']['lzma']['size']))

    print(f"\nWin counts: gzip={gzip_wins}, lzma={lzma_wins}, Nacrith={nacrith_wins}")

    # Shutdown compressor
    compressor.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
