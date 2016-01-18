#include "convnetlib.h"
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>

static PyMethodDef convnetlib_funcs[] = {
    { "conv_forward", (PyCFunction) conv_forward, METH_VARARGS, NULL },
    { "conv_prev_layer_delta", (PyCFunction) conv_prev_layer_delta, METH_VARARGS, NULL },
    { "conv_backward", (PyCFunction) conv_backward, METH_VARARGS, NULL },
    { "__test", (PyCFunction) __test, METH_VARARGS, NULL },
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

PyObject* __test(PyObject* self, PyObject* args) {
    PyObject* out_array;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &out_array
                                               )) {
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

PyObject* conv_forward(PyObject* self, PyObject* args) {
    PyObject* X_array;
    PyObject* w;
    PyObject* b;
    int stride;
    PyObject* out_array;

    /*  parse numpy array argument and value */
    if (!PyArg_ParseTuple(args, "O!O!O!iO!", &PyArray_Type, &X_array,
                                             &PyList_Type, &w,  // lsit of numpy arrays
                                             &PyList_Type, &b,  // list of numbers
                                             &stride,
                                             &PyArray_Type, &out_array
                                             )) {
        return NULL;
    }
    
    int filters_count_w = PyList_Size(w);
    int filters_count_b = PyList_Size(b);
    
    
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
    
    npy_intp out_shape[] = {0, 0, 0};
    double*** out = NULL;
    if (c_array_from_pyarray3d(out_array, &out, (npy_intp*) out_shape) < 0) {
        PyErr_SetString(PyExc_TypeError, "Error converting out to c array");
        return NULL;
    }
    
    double*** w_f = NULL;
    npy_intp w_f_shape[] = {0, 0, 0};
    for (int f = 0; f < filters_count_w; ++f) {
        PyObject* obj_w_f = PyList_GetItem(w, f);
        PyObject* obj_b_f = PyList_GetItem(b, f);
        
        double b_f = PyFloat_AsDouble(obj_b_f);
        
        if (c_array_from_pyarray3d(obj_w_f, &w_f, (npy_intp*) w_f_shape) < 0) {
            PyErr_SetString(PyExc_TypeError, "Error converting w to c array");
            return NULL;
        }
        
        for (int x = 0; x < out_shape[0]; ++x) {
            for (int y = 0; y < out_shape[1]; ++y) {
                
                double conv = 0.0;
                
                for (int i = 0; i < w_f_shape[0]; ++i) {
                    for (int j = 0; j < w_f_shape[1]; ++j) {
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
}


PyObject* conv_prev_layer_delta(PyObject* self, PyObject* args) {
    PyObject* current_layer_delta_array;
    PyObject* w;
    PyObject* out_array;

    /*  parse numpy array argument and value */
    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &current_layer_delta_array,
                                          &PyList_Type, &w,  // lsit of numpy arrays
                                          &PyArray_Type, &out_array
                                          )) {
        return NULL;
    }
    
    npy_intp current_layer_delta_shape[] = {0, 0, 0};
    double*** current_layer_delta = NULL;
    if (c_array_from_pyarray3d(current_layer_delta_array, &current_layer_delta, (npy_intp*) current_layer_delta_shape) < 0) {
        PyErr_SetString(PyExc_TypeError, "Error converting current_layer_delta to c array");
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
    for (int f = 0; f < current_layer_delta_shape[2]; ++f) {
        PyObject* obj_w_f = PyList_GetItem(w, f);
        
        if (c_array_from_pyarray3d(obj_w_f, &w_f, (npy_intp*) w_f_shape) < 0) {
            PyErr_SetString(PyExc_TypeError, "Error converting w to c array");
            return NULL;
        }
        // fprintf(stderr, "w[%d] shape: %ld %ld %ld\n", f, w_f_shape[0], w_f_shape[1], w_f_shape[2]);
        
        for (int x = 0; x < out_shape[0]; ++x) {
            for (int y = 0; y < out_shape[1]; ++y) {
                for (int z = 0; z < out_shape[2]; ++z) {
                    double conv = 0.0;
                    
                    for (int i = 0; i < current_layer_delta_shape[0]; ++i) {
                        if ((x - i) >= 0 && (x - i) < w_f_shape[0]) {
                        
                            for (int j = 0; j < current_layer_delta_shape[1]; ++j) {
                                if ((y - j) >= 0 && (y - j) < w_f_shape[1]) {
                                    conv += current_layer_delta[i][j][f] * w_f[x - i][y - j][z];
                                }
                            }
                        }
                    }
                    
                    out[x][y][z] = conv;
                }
            }
        }
    }
    return out_array;
}


PyObject* conv_backward(PyObject* self, PyObject* args) {
    PyObject* current_layer_delta_array;
    PyObject* prev_layer_out_array;
    PyObject* dw;
    PyObject* db;

    /*  parse numpy array argument and value */
    if (!PyArg_ParseTuple(args, "O!O!O!O!", &PyArray_Type, &current_layer_delta_array,
                                            &PyArray_Type, &prev_layer_out_array,
                                            &PyList_Type, &dw,  // lsit of numpy arrays
                                            &PyList_Type, &db   // lsit of numpy arrays
                                            )) {
        return NULL;
    }
    
    npy_intp current_layer_delta_shape[] = {0, 0, 0};
    double*** current_layer_delta = NULL;
    if (c_array_from_pyarray3d(current_layer_delta_array, &current_layer_delta, (npy_intp*) current_layer_delta_shape) < 0) {
        PyErr_SetString(PyExc_TypeError, "Error converting current_layer_delta to c array");
        return NULL;
    }
    
    npy_intp prev_layer_out_shape[] = {0, 0, 0};
    double*** prev_layer_out = NULL;
    if (c_array_from_pyarray3d(prev_layer_out_array, &prev_layer_out, (npy_intp*) prev_layer_out_shape) < 0) {
        PyErr_SetString(PyExc_TypeError, "Error converting prev_layer_out to c array");
        return NULL;
    }
    
    int filters_count_dw = PyList_Size(dw);
    int filters_count_db = PyList_Size(db);
    
    if (filters_count_dw != filters_count_db || filters_count_dw < 0) {
        PyErr_SetString(PyExc_ValueError, "Sizes of dw and db have to be the same and non-negative");
        return NULL;
    }
    
    double conv = 0.0;
    double*** dw_f = NULL;
    npy_intp dw_f_shape[] = {0, 0, 0};
    for (int f = 0; f < current_layer_delta_shape[2]; ++f) {
        PyObject* obj_dw_f = PyList_GetItem(dw, f);
        PyObject* obj_db_f = PyList_GetItem(db, f);
        
        if (c_array_from_pyarray3d(obj_dw_f, &dw_f, (npy_intp*) dw_f_shape) < 0) {
            PyErr_SetString(PyExc_TypeError, "Error converting dw to c array");
            return NULL;
        }
        
        // calc dE/dW
        for (int x = 0; x < dw_f_shape[0]; ++x) {
            for (int y = 0; y < dw_f_shape[1]; ++y) {
                for (int z = 0; z < dw_f_shape[2]; ++z) {
                    conv = 0.0;
                    for (int i = 0; i < current_layer_delta_shape[0]; ++i) {
                        for (int j = 0; j < current_layer_delta_shape[1]; ++j) {
                            conv += current_layer_delta[i][j][f] * prev_layer_out[i + x][j + y][z];
                        }
                    }
                    
                    dw_f[x][y][z] += conv;
                }
            }
        }
        
        // calc dE/db
        conv = 0.0;
        for (int i = 0; i < current_layer_delta_shape[0]; ++i) {
            for (int j = 0; j < current_layer_delta_shape[1]; ++j) {
                conv += current_layer_delta[i][j][f];
            }
        }
        
        obj_db_f = PyNumber_InPlaceAdd(obj_db_f, PyFloat_FromDouble(conv));
    }
    Py_RETURN_NONE;
}
