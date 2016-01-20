#define PY_ARRAY_UNIQUE_SYMBOL convnetlib_ARRAY_API

#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>

#include "conv.h"
#include "pool.h"
#include "util.h"

static PyObject* __test(PyObject* self, PyObject* args);

PyMethodDef convnetlib_funcs[] = {
    { "conv_forward", (PyCFunction) conv_forward, METH_VARARGS, NULL },
    { "conv_prev_layer_delta", (PyCFunction) conv_prev_layer_delta, METH_VARARGS, NULL },
    { "conv_backward", (PyCFunction) conv_backward, METH_VARARGS, NULL },
    
    { "__test", (PyCFunction) __test, METH_VARARGS, NULL },
    { NULL, NULL, 0, NULL }
};

PyMODINIT_FUNC
initconvnetlib(void) {
    Py_InitModule3("convnetlib", convnetlib_funcs, "convnetlib - library for convnet python module");
    import_array();
}




PyObject* __test(PyObject* self, PyObject* args) {
   PyObject* out_array;
   if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &out_array)) {
       return NULL;
   }

   npy_intp out_shape[] = {0, 0, 0};
   double*** out = NULL;
   if (c_array_from_pyarray3d(out_array, &out, (npy_intp*) out_shape) < 0) {
       PyErr_SetString(PyExc_TypeError, "Error converting out to c array");
       return NULL;
   }

   for (int i = 0; i < out_shape[0]; ++i) {
       for (int j = 0; j < out_shape[1]; ++j) {
           for (int k = 0; k < out_shape[2]; ++k) {
               out[i][j][k] = 1;
           }
       }
   }

   return out_array;
}

