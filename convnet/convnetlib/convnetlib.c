#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL convnetlib_ARRAY_API

#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>

#include "conv.h"
#include "pool.h"
#include "util.h"

PyMethodDef convnetlib_funcs[] = {
    { "conv_forward", (PyCFunction) conv_forward, METH_VARARGS, NULL },
    { "conv_backward", (PyCFunction) conv_backward, METH_VARARGS, NULL },
    { "conv_prev_layer_delta", (PyCFunction) conv_prev_layer_delta, METH_VARARGS, NULL },
    
    { "pool_forward", (PyCFunction) pool_forward, METH_VARARGS, NULL },
    { "pool_prev_layer_delta", (PyCFunction) pool_prev_layer_delta, METH_VARARGS, NULL },
    { NULL, NULL, 0, NULL }
};

PyMODINIT_FUNC
initconvnetlib(void) {
    Py_InitModule3("convnetlib", convnetlib_funcs, "convnetlib - library for convnet python module");
    import_array();
}

