# Makefile for Nacrith v7.6
# Builds the C arithmetic coder extension

PYTHON := python3
SETUP := setup.py

.PHONY: all build clean install test help

all: build

# Build the C extension in-place
build:
	@echo "Building arithmetic_coder C extension..."
	$(PYTHON) $(SETUP) build_ext --inplace
	@echo "✓ Build complete"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -f arithmetic_coder*.so
	rm -f *.pyc
	rm -rf __pycache__/
	rm -rf *.egg-info/
	@echo "✓ Clean complete"

# Install dependencies
install:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"

# Build + install (full setup)
setup: install build
	@echo "✓ Setup complete - Nacrith v7.6 is ready!"

# Run tests
test: build
	@echo "Testing compression/decompression..."
	@echo "Test 1: Compress sample file"
	$(PYTHON) nacrith.py c ../v7_4/web_corpus/json_schema.json /tmp/test.nc5
	@echo "Test 2: Decompress"
	$(PYTHON) nacrith.py d /tmp/test.nc5 /tmp/test_restored.json
	@echo "Test 3: Verify lossless"
	@diff ../v7_4/web_corpus/json_schema.json /tmp/test_restored.json && echo "✓ Lossless roundtrip PASS" || echo "✗ Roundtrip FAILED"
	@rm -f /tmp/test.nc5 /tmp/test_restored.json

# Show help
help:
	@echo "Nacrith v7.6 - Makefile targets:"
	@echo ""
	@echo "  make build    - Build C extension (default)"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make install  - Install Python dependencies"
	@echo "  make setup    - Full setup (install + build)"
	@echo "  make test     - Run compression tests"
	@echo "  make help     - Show this help"
	@echo ""
	@echo "Quick start:"
	@echo "  1. make setup    # Install deps + build"
	@echo "  2. make test     # Verify it works"
