#include "util.h"

int c_array_from_pyarray2d(PyObject* arr, double*** out, npy_intp* out_dims) {
    //Create C arrays from numpy objects:
    PyArray_Descr *descr = PyArray_DescrFromType(NPY_DOUBLE);
    if (PyArray_AsCArray(&arr, (void**) out, out_dims, 2, descr) < 0) {
        PyErr_SetString(PyExc_TypeError, "error converting to c array");
        return -1;
    }
    return 0;
}

int c_array_from_pyarray3d(PyObject* arr, double**** out, npy_intp* out_dims) {
    //Create C arrays from numpy objects:
    PyArray_Descr *descr = PyArray_DescrFromType(NPY_DOUBLE);
    if (PyArray_AsCArray(&arr, (void***) out, out_dims, 3, descr) < 0) {
        return -1;
    }
    return 0;
}
