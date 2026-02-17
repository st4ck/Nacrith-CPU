# Installation Guide

## Prerequisites

- **Python 3.8+**
- **GCC compiler** (for building C extension)
- **4 GB+ RAM** (for loading trigram tables)
- **~500 MB disk space** (for trigram tables and dependencies)

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/st4ck/Nacrith-CPU.git
cd Nacrith-CPU
```

### 2. Install Dependencies & Build

```bash
make setup
```

This will:
1. Install Python dependencies from `requirements.txt`
2. Build the C arithmetic coder extension
3. Verify the installation

**OR** step by step:

```bash
# Install Python packages
pip install -r requirements.txt

# Build C extension
python setup.py build_ext --inplace

# Verify
python -c "import arithmetic_coder; print('✓ C extension loaded')"
```

### 3. Download or Build Trigram Tables

**Option A: Download prebuilt table** (recommended)

Download `trigram_en.npz` (88 MB) from the [Releases page](https://github.com/st4ck/Nacrith-CPU/releases) and place it in the `trigrams/` directory.

**Option B: Build from WikiText-103**

```bash
python build_table.py --corpus wikitext-103 --output trigrams/trigram_en.npz
```

This will:
- Download WikiText-103 corpus (~500 MB)
- Tokenize with SmolLM2-135M tokenizer
- Count n-gram frequencies
- Build sparse top-K tables
- Save compressed `.npz` file (88 MB)

**Note:** Building takes ~30-60 minutes and requires ~8 GB RAM.

### 4. Test the Installation

```bash
make test
```

Or manually:

```bash
# Create a test file
echo "The quick brown fox jumps over the lazy dog." > test.txt

# Compress
python nacrith.py c test.txt test.nc5

# Decompress
python nacrith.py d test.nc5 restored.txt

# Verify
diff test.txt restored.txt && echo "✓ Lossless compression verified"
```

## System-Specific Notes

### Linux (Ubuntu/Debian)

```bash
# Install build tools
sudo apt-get update
sudo apt-get install build-essential python3-dev

# Install Python dependencies
pip3 install -r requirements.txt

# Build
python3 setup.py build_ext --inplace
```

### macOS

```bash
# Install Xcode command line tools (if not already installed)
xcode-select --install

# Install Python dependencies
pip3 install -r requirements.txt

# Build
python3 setup.py build_ext --inplace
```

### Windows (WSL recommended)

Nacrith CPU uses Unix-specific features (multiprocessing with fork, shared memory). **Use WSL (Windows Subsystem for Linux)** for the best experience.

In WSL:

```bash
sudo apt-get update
sudo apt-get install build-essential python3-dev
pip3 install -r requirements.txt
python3 setup.py build_ext --inplace
```

## Troubleshooting

### "arithmetic_coder.c: No such file or directory"

Ensure you're in the correct directory:

```bash
ls -la arith_coder.c setup.py
```

### "Python.h: No such file or directory"

Install Python development headers:

```bash
# Ubuntu/Debian
sudo apt-get install python3-dev

# macOS (usually included with Xcode tools)
xcode-select --install
```

### "fatal error: 'numpy/arrayobject.h'"

Update numpy:

```bash
pip install --upgrade numpy
```

### "ModuleNotFoundError: No module named 'transformers'"

Install missing dependencies:

```bash
pip install -r requirements.txt
```

### Build fails with "error: invalid use of '__int128'"

Your compiler doesn't support `__int128`. This is rare on modern systems. Edit `arith_coder.c` to use 64-bit arithmetic (with reduced precision).

### "SharedMemory: [Errno 28] No space left on device"

You're running low on `/dev/shm` space. Check:

```bash
df -h /dev/shm
```

If it's full, clean up:

```bash
ls -la /dev/shm
rm /dev/shm/psm_*  # Remove orphaned shared memory segments
```

Or increase the size (requires root):

```bash
sudo mount -o remount,size=4G /dev/shm
```

## Verifying the Installation

Run the full test suite:

```bash
# Quick test (no table needed)
python -c "from arithmetic_coder import ArithmeticEncoder, ArithmeticDecoder; print('✓ C extension works')"

# Full compression test (requires trigram table)
python nacrith.py benchmark README.md
```

Expected output:
```
Loading 1 tables into shared memory...
Ready: 1 tables | Names: en | Workers: 2 processes
Compressing README.md...
  Original: 23.3 KB
  Compressed: 6.2 KB (26.6%)
  Ratio: 2.13 bpb
✓ Verified lossless
```

## Next Steps

- Read [README.md](README.md) for usage examples
- See [BENCHMARK_SUMMARY.md](BENCHMARK_SUMMARY.md) for performance analysis
- Build custom tables with `build_table.py` for your specific domain

## Getting Help

- **Issues:** [GitHub Issues](https://github.com/st4ck/Nacrith-CPU/issues)
- **Documentation:** See README.md and inline code comments
- **Performance:** Check BENCHMARK_SUMMARY.md for expected results
