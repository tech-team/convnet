#include <Python.h>

static PyObject* conv_forward(PyObject* self, PyObject* args);
static PyObject* conv_backward(PyObject* self, PyObject* args);
static PyObject* conv_prev_layer_delta(PyObject* self, PyObject* args);
static PyObject* __test(PyObject* self, PyObject* args);
