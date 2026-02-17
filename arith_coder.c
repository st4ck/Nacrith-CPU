/*
 * Fast arithmetic encoder/decoder as a CPython C extension.
 * Drop-in replacement for arithmetic_coder.py with identical semantics.
 *
 * Build: python setup.py build_ext --inplace
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>
#include <stdlib.h>
#include <string.h>

/* ---- constants ---- */
#define PRECISION  32
#define FULL       ((uint64_t)1 << PRECISION)
#define HALF       ((uint64_t)1 << (PRECISION - 1))
#define QUARTER    ((uint64_t)1 << (PRECISION - 2))
#define MAX_RANGE  (FULL - 1)

/* ---- dynamic bit buffer ---- */
typedef struct {
    uint8_t *data;
    size_t   len;
    size_t   cap;
} BitBuf;

static void bitbuf_init(BitBuf *b) {
    b->cap = 4096;
    b->data = (uint8_t *)malloc(b->cap);
    b->len = 0;
}

static void bitbuf_push(BitBuf *b, uint8_t bit) {
    if (b->len >= b->cap) {
        b->cap *= 2;
        b->data = (uint8_t *)realloc(b->data, b->cap);
    }
    b->data[b->len++] = bit;
}

static void bitbuf_free(BitBuf *b) {
    free(b->data);
    b->data = NULL;
    b->len = b->cap = 0;
}

/* ==================================================================
 * ArithmeticEncoder
 * ================================================================== */
typedef struct {
    PyObject_HEAD
    uint64_t low;
    uint64_t high;
    int64_t  pending_bits;
    BitBuf   bits;
} EncoderObject;

static void encoder_output_bit(EncoderObject *self, uint8_t bit) {
    bitbuf_push(&self->bits, bit);
    while (self->pending_bits > 0) {
        bitbuf_push(&self->bits, 1 - bit);
        self->pending_bits--;
    }
}

static int Encoder_init(EncoderObject *self, PyObject *args, PyObject *kw) {
    (void)args; (void)kw;
    self->low = 0;
    self->high = MAX_RANGE;
    self->pending_bits = 0;
    bitbuf_init(&self->bits);
    return 0;
}

static void Encoder_dealloc(EncoderObject *self) {
    bitbuf_free(&self->bits);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

/*
 * encode_symbol(cdf: list[int], symbol_index: int)
 * cdf is a Python list, we read it via PyList fast access.
 */
static PyObject *Encoder_encode_symbol(EncoderObject *self, PyObject *args) {
    PyObject *cdf_list;
    int symbol_index;

    if (!PyArg_ParseTuple(args, "O!i", &PyList_Type, &cdf_list, &symbol_index))
        return NULL;

    Py_ssize_t cdf_len = PyList_GET_SIZE(cdf_list);
    if (symbol_index < 0 || symbol_index + 1 >= cdf_len) {
        PyErr_SetString(PyExc_IndexError, "symbol_index out of range");
        return NULL;
    }

    int64_t total = PyLong_AsLongLong(PyList_GET_ITEM(cdf_list, cdf_len - 1));
    int64_t cdf_lo = PyLong_AsLongLong(PyList_GET_ITEM(cdf_list, symbol_index));
    int64_t cdf_hi = PyLong_AsLongLong(PyList_GET_ITEM(cdf_list, symbol_index + 1));

    uint64_t rng = self->high - self->low + 1;
    self->high = self->low + (uint64_t)(((__int128)rng * cdf_hi) / total) - 1;
    self->low  = self->low + (uint64_t)(((__int128)rng * cdf_lo) / total);

    for (;;) {
        if (self->high < HALF) {
            encoder_output_bit(self, 0);
            self->low  = self->low << 1;
            self->high = (self->high << 1) | 1;
        } else if (self->low >= HALF) {
            encoder_output_bit(self, 1);
            self->low  = (self->low - HALF) << 1;
            self->high = ((self->high - HALF) << 1) | 1;
        } else if (self->low >= QUARTER && self->high < 3 * QUARTER) {
            self->pending_bits++;
            self->low  = (self->low - QUARTER) << 1;
            self->high = ((self->high - QUARTER) << 1) | 1;
        } else {
            break;
        }
    }

    self->low  &= MAX_RANGE;
    self->high &= MAX_RANGE;

    Py_RETURN_NONE;
}

static PyObject *Encoder_finish(EncoderObject *self, PyObject *Py_UNUSED(args)) {
    self->pending_bits++;
    if (self->low < QUARTER)
        encoder_output_bit(self, 0);
    else
        encoder_output_bit(self, 1);

    /* pad to byte boundary */
    while (self->bits.len % 8 != 0)
        bitbuf_push(&self->bits, 0);

    /* pack bits into bytes */
    size_t nbytes = self->bits.len / 8;
    PyObject *result = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)nbytes);
    if (!result) return NULL;
    char *out = PyBytes_AS_STRING(result);
    for (size_t i = 0; i < nbytes; i++) {
        uint8_t byte = 0;
        size_t base = i * 8;
        for (int j = 0; j < 8; j++)
            byte = (byte << 1) | self->bits.data[base + j];
        out[i] = (char)byte;
    }
    return result;
}

static PyObject *Encoder_get_bit_count(EncoderObject *self, PyObject *Py_UNUSED(args)) {
    return PyLong_FromLongLong((long long)(self->bits.len + self->pending_bits));
}

static PyMethodDef Encoder_methods[] = {
    {"encode_symbol", (PyCFunction)Encoder_encode_symbol, METH_VARARGS, "Encode a symbol."},
    {"finish",        (PyCFunction)Encoder_finish,        METH_NOARGS,  "Finalize and return bytes."},
    {"get_bit_count", (PyCFunction)Encoder_get_bit_count, METH_NOARGS,  "Return bits written so far."},
    {NULL}
};

static PyTypeObject EncoderType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "arithmetic_coder.ArithmeticEncoder",
    .tp_basicsize = sizeof(EncoderObject),
    .tp_flags     = Py_TPFLAGS_DEFAULT,
    .tp_new       = PyType_GenericNew,
    .tp_init      = (initproc)Encoder_init,
    .tp_dealloc   = (destructor)Encoder_dealloc,
    .tp_methods   = Encoder_methods,
};

/* ==================================================================
 * ArithmeticDecoder
 * ================================================================== */
typedef struct {
    PyObject_HEAD
    uint8_t *bits;
    size_t   bits_len;
    size_t   bit_pos;
    uint64_t low;
    uint64_t high;
    uint64_t value;
} DecoderObject;

static uint8_t decoder_read_bit(DecoderObject *self) {
    if (self->bit_pos < self->bits_len) {
        return self->bits[self->bit_pos++];
    }
    return 0;
}

static int Decoder_init(DecoderObject *self, PyObject *args, PyObject *kw) {
    (void)kw;
    Py_buffer buf;
    if (!PyArg_ParseTuple(args, "y*", &buf))
        return -1;

    /* unpack bytes into bit array */
    size_t nbits = (size_t)buf.len * 8;
    self->bits = (uint8_t *)malloc(nbits);
    if (!self->bits) {
        PyBuffer_Release(&buf);
        PyErr_NoMemory();
        return -1;
    }
    self->bits_len = nbits;
    const uint8_t *src = (const uint8_t *)buf.buf;
    for (Py_ssize_t i = 0; i < buf.len; i++) {
        uint8_t byte = src[i];
        size_t base = (size_t)i * 8;
        for (int j = 7; j >= 0; j--)
            self->bits[base + (7 - j)] = (byte >> j) & 1;
    }
    PyBuffer_Release(&buf);

    self->bit_pos = 0;
    self->low  = 0;
    self->high = MAX_RANGE;
    self->value = 0;
    for (int i = 0; i < PRECISION; i++)
        self->value = (self->value << 1) | decoder_read_bit(self);

    return 0;
}

static void Decoder_dealloc(DecoderObject *self) {
    free(self->bits);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *Decoder_decode_symbol(DecoderObject *self, PyObject *args) {
    PyObject *cdf_list;
    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &cdf_list))
        return NULL;

    Py_ssize_t cdf_len = PyList_GET_SIZE(cdf_list);
    int64_t total = PyLong_AsLongLong(PyList_GET_ITEM(cdf_list, cdf_len - 1));
    uint64_t rng = self->high - self->low + 1;

    int64_t scaled_value = (int64_t)((((__int128)(self->value - self->low + 1)) * total - 1) / rng);

    /* binary search for symbol */
    Py_ssize_t lo = 0, hi = cdf_len - 2;
    while (lo <= hi) {
        Py_ssize_t mid = (lo + hi) / 2;
        int64_t cdf_mid1 = PyLong_AsLongLong(PyList_GET_ITEM(cdf_list, mid + 1));
        if (cdf_mid1 <= scaled_value)
            lo = mid + 1;
        else
            hi = mid - 1;
    }
    Py_ssize_t symbol = lo;

    int64_t cdf_lo_val = PyLong_AsLongLong(PyList_GET_ITEM(cdf_list, symbol));
    int64_t cdf_hi_val = PyLong_AsLongLong(PyList_GET_ITEM(cdf_list, symbol + 1));

    self->high = self->low + (uint64_t)(((__int128)rng * cdf_hi_val) / total) - 1;
    self->low  = self->low + (uint64_t)(((__int128)rng * cdf_lo_val) / total);

    for (;;) {
        if (self->high < HALF) {
            self->low   = self->low << 1;
            self->high  = (self->high << 1) | 1;
            self->value = (self->value << 1) | decoder_read_bit(self);
        } else if (self->low >= HALF) {
            self->low   = (self->low - HALF) << 1;
            self->high  = ((self->high - HALF) << 1) | 1;
            self->value = ((self->value - HALF) << 1) | decoder_read_bit(self);
        } else if (self->low >= QUARTER && self->high < 3 * QUARTER) {
            self->low   = (self->low - QUARTER) << 1;
            self->high  = ((self->high - QUARTER) << 1) | 1;
            self->value = ((self->value - QUARTER) << 1) | decoder_read_bit(self);
        } else {
            break;
        }
    }

    self->low   &= MAX_RANGE;
    self->high  &= MAX_RANGE;
    self->value &= MAX_RANGE;

    return PyLong_FromSsize_t(symbol);
}

static PyMethodDef Decoder_methods[] = {
    {"decode_symbol", (PyCFunction)Decoder_decode_symbol, METH_VARARGS, "Decode a symbol."},
    {NULL}
};

static PyTypeObject DecoderType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "arithmetic_coder.ArithmeticDecoder",
    .tp_basicsize = sizeof(DecoderObject),
    .tp_flags     = Py_TPFLAGS_DEFAULT,
    .tp_new       = PyType_GenericNew,
    .tp_init      = (initproc)Decoder_init,
    .tp_dealloc   = (destructor)Decoder_dealloc,
    .tp_methods   = Decoder_methods,
};

/* ==================================================================
 * Module definition
 * ================================================================== */
static PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "arithmetic_coder",
    "Fast arithmetic encoder/decoder in C.",
    -1,
    NULL
};

PyMODINIT_FUNC PyInit_arithmetic_coder(void) {
    PyObject *m;

    if (PyType_Ready(&EncoderType) < 0) return NULL;
    if (PyType_Ready(&DecoderType) < 0) return NULL;

    m = PyModule_Create(&moduledef);
    if (!m) return NULL;

    Py_INCREF(&EncoderType);
    PyModule_AddObject(m, "ArithmeticEncoder", (PyObject *)&EncoderType);
    Py_INCREF(&DecoderType);
    PyModule_AddObject(m, "ArithmeticDecoder", (PyObject *)&DecoderType);

    return m;
}
