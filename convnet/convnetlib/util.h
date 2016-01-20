#ifndef _CONVNETLIB_UTIL_H_
#define _CONVNETLIB_UTIL_H_

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL convnetlib_ARRAY_API

#include <Python.h>
#include <numpy/arrayobject.h>

int c_array_from_pyarray2d(PyObject** arr, double*** out, npy_intp* out_dims);
int c_array_from_pyarray3d(PyObject** arr, double**** out, npy_intp* out_dims);

#endif // _CONVNETLIB_UTIL_H_
