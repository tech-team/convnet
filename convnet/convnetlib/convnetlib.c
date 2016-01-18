#include "convnetlib.h"
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>

static PyMethodDef convnetlib_funcs[] = {
    { "conv_forward", (PyCFunction) conv_forward, METH_VARARGS, NULL },
    { NULL, NULL, 0, NULL }
};

PyMODINIT_FUNC
initconvnetlib(void) {
    Py_InitModule3("convnetlib", convnetlib_funcs, "Extension module example!");
    import_array();
}


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

PyObject* conv_forward(PyObject* self, PyObject* args) {
    PyObject* X_array;
    PyObject* w;
    PyObject* b;
    int stride;
    int filter_size;
    PyObject* out_array;

    /*  parse numpy array argument and value */
    if (!PyArg_ParseTuple(args, "O!O!O!iiO!", &PyArray_Type, &X_array,
                                               &PyList_Type, &w,  // lsit of numpy arrays
                                               &PyList_Type, &b,  // list of numbers
                                               &stride,
                                               &filter_size,
                                               &PyArray_Type, &out_array
                                               )) {
        return NULL;
    }
    
    int filters_count_w = PyList_Size(w);
    int filters_count_b = PyList_Size(w);
    
    
    if (filters_count_w != filters_count_b || filters_count_w < 0) {
        PyErr_SetString(PyExc_ValueError, "Sizes of w and b have to be the same");
        return NULL;
    }
    
    npy_intp x_shape[] = {0, 0, 0};
    double*** X = NULL;
    if (c_array_from_pyarray3d(X_array, &X, (npy_intp*) x_shape) < 0) {
        PyErr_SetString(PyExc_TypeError, "Error converting X to c array");
        return NULL;
    }
    // fprintf(stderr, "X shape: %ld %ld %ld\n", x_shape[0], x_shape[1], x_shape[2]);
    
    npy_intp out_shape[] = {0, 0, 0};
    double*** out = NULL;
    if (c_array_from_pyarray3d(out_array, &out, (npy_intp*) out_shape) < 0) {
        PyErr_SetString(PyExc_TypeError, "Error converting out to c array");
        return NULL;
    }
    // fprintf(stderr, "out shape: %ld %ld %ld\n", out_shape[0], out_shape[1], out_shape[2]);
    
    double*** w_f = NULL;
    npy_intp w_f_shape[] = {0, 0, 0};
    for (int f = 0; f < filters_count_w; ++f) {
        PyObject* obj_w_f = PyList_GetItem(w, f);
        PyObject* obj_b_f = PyList_GetItem(b, f);
        
        // obj_w_f = PyArray_FromAny(obj_w_f, NPY_DOUBLE, 3, 3);
        double b_f = PyFloat_AsDouble(obj_b_f);
        
        if (c_array_from_pyarray3d(obj_w_f, &w_f, (npy_intp*) w_f_shape) < 0) {
            PyErr_SetString(PyExc_TypeError, "Error converting w to c array");
            return NULL;
        }
        
        // fprintf(stderr, "w[%d] shape: %ld %ld %ld\n", f, w_f_shape[0], w_f_shape[1], w_f_shape[2]);
        
        for (int x = 0; x < out_shape[0]; ++x) {
            for (int y = 0; y < out_shape[1]; ++y) {
                
                double conv = 0.0;
                
                for (int i = 0; i < filter_size; ++i) {
                    for (int j = 0; j < filter_size; ++j) {
                        for (int z = 0; z < x_shape[2]; ++z) {
                            conv += X[stride * x + i][stride * y + j][z] * w_f[i][j][z];
                        }
                    }
                }
                
                out[x][y][f] = b_f + conv;
            }
        }
    }
    return out_array;

    /*  in case bad things happen */
fail:
    return NULL;
}
