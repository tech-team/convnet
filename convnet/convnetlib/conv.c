#include "conv.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL convnetlib_ARRAY_API
#include <numpy/arrayobject.h>

#include "util.h"

PyObject* conv_forward(PyObject* self, PyObject* args) {
    PyObject* X_array;
    PyObject* w;
    PyObject* b;
    int stride;
    PyObject* out_array;

    /*  parse numpy array argument and value */
    if (!PyArg_ParseTuple(args, "O!O!O!iO!", &PyArray_Type, &X_array,
                                             &PyList_Type, &w,  /* list of numpy arrays */
                                             &PyList_Type, &b,  /* list of numbers */
                                             &stride,
                                             &PyArray_Type, &out_array
                                             )) {
        return NULL;
    }
    
    Py_ssize_t filters_count_w, filters_count_b;
    filters_count_w = PyList_Size(w);
    filters_count_b = PyList_Size(b);
    
    if (filters_count_w != filters_count_b || filters_count_w < 0) {
        PyErr_SetString(PyExc_ValueError, "Sizes of w and b have to be the same");
        return NULL;
    }
    
    npy_intp x_shape[] = {0, 0, 0};
    double*** X = NULL;
    if (c_array_from_pyarray3d(&X_array, &X, (npy_intp*) x_shape) < 0) {
        goto fail;
    }
    
    npy_intp out_shape[] = {0, 0, 0};
    double*** out = NULL;
    if (c_array_from_pyarray3d(&out_array, &out, (npy_intp*) out_shape) < 0) {
        goto fail;
    }
    
    PyObject* obj_w_f = NULL;
    PyObject* obj_b_f = NULL;
    npy_intp w_f_shape[] = {0, 0, 0};
    double*** w_f = NULL;
    double b_f;
    
    int f, x, y,
        i, j, z;
    for (f = 0; f < filters_count_w; ++f) {
        obj_w_f = PyList_GetItem(w, f);
        obj_b_f = PyList_GetItem(b, f);
        
        b_f = PyFloat_AsDouble(obj_b_f);
        
        if (c_array_from_pyarray3d(&obj_w_f, &w_f, (npy_intp*) w_f_shape) < 0) {
            goto fail;
        }
        
        for (x = 0; x < out_shape[0]; ++x) {
            for (y = 0; y < out_shape[1]; ++y) {
                
                double conv = 0.0;
                
                for (i = 0; i < w_f_shape[0]; ++i) {
                    for (j = 0; j < w_f_shape[1]; ++j) {
                        for (z = 0; z < x_shape[2]; ++z) {
                            conv += X[stride * x + i][stride * y + j][z] * w_f[i][j][z];
                        }
                    }
                }
                
                out[x][y][f] = b_f + conv;
            }
        }
        
        PyArray_Free(obj_w_f, w_f);
        w_f = NULL;
    }
    
    PyArray_Free(X_array, X);
    PyArray_Free(out_array, out);
    Py_RETURN_NONE;
    
fail:
    PyArray_Free(X_array, X);
    PyArray_Free(out_array, out);
    return NULL;
}


PyObject* conv_backward(PyObject* self, PyObject* args) {
    PyObject* current_layer_delta_array;
    PyObject* prev_layer_out_array;
    PyObject* dw;
    PyObject* db;

    /*  parse numpy array argument and value */
    if (!PyArg_ParseTuple(args, "O!O!O!O!", &PyArray_Type, &current_layer_delta_array,
                                            &PyArray_Type, &prev_layer_out_array,
                                            &PyList_Type, &dw,  /* list of numpy arrays */
                                            &PyList_Type, &db   /* list of numpy arrays */
                                            )) {
        return NULL;
    }
    
    Py_ssize_t filters_count_dw, filters_count_db;
    filters_count_dw = PyList_Size(dw);
    filters_count_db = PyList_Size(db);
    
    if (filters_count_dw != filters_count_db || filters_count_dw < 0) {
        PyErr_SetString(PyExc_ValueError, "Sizes of dw and db have to be the same and non-negative");
        return NULL;
    }
    
    
    npy_intp current_layer_delta_shape[] = {0, 0, 0};
    double*** current_layer_delta = NULL;
    if (c_array_from_pyarray3d(&current_layer_delta_array, &current_layer_delta, (npy_intp*) current_layer_delta_shape) < 0) {
        goto fail;
    }
    
    npy_intp prev_layer_out_shape[] = {0, 0, 0};
    double*** prev_layer_out = NULL;
    if (c_array_from_pyarray3d(&prev_layer_out_array, &prev_layer_out, (npy_intp*) prev_layer_out_shape) < 0) {
        goto fail;
    }
    
    
    PyObject* obj_dw_f = NULL;
    PyObject* obj_db_f = NULL;
    double conv = 0.0;
    double*** dw_f = NULL;
    npy_intp dw_f_shape[] = {0, 0, 0};
    
    int f, x, y,
        z, i, j;
    for (f = 0; f < current_layer_delta_shape[2]; ++f) {
        obj_dw_f = PyList_GetItem(dw, f);
        obj_db_f = PyList_GetItem(db, f);
        
        if (c_array_from_pyarray3d(&obj_dw_f, &dw_f, (npy_intp*) dw_f_shape) < 0) {
            goto fail;
        }
        
        /* calc dE/dW */
        for (x = 0; x < dw_f_shape[0]; ++x) {
            for (y = 0; y < dw_f_shape[1]; ++y) {
                for (z = 0; z < dw_f_shape[2]; ++z) {
                    conv = 0.0;
                    for (i = 0; i < current_layer_delta_shape[0]; ++i) {
                        for (j = 0; j < current_layer_delta_shape[1]; ++j) {
                            conv += current_layer_delta[i][j][f] * prev_layer_out[i + x][j + y][z];
                        }
                    }
                    
                    dw_f[x][y][z] += conv;
                }
            }
        }
        
        /* calc dE/db */
        conv = 0.0;
        for (i = 0; i < current_layer_delta_shape[0]; ++i) {
            for (j = 0; j < current_layer_delta_shape[1]; ++j) {
                conv += current_layer_delta[i][j][f];
            }
        }
        
        obj_db_f = PyNumber_InPlaceAdd(obj_db_f, PyFloat_FromDouble(conv));
        
        PyArray_Free(obj_dw_f, dw_f);
        dw_f = NULL;
    }
    PyArray_Free(current_layer_delta_array, current_layer_delta);
    PyArray_Free(prev_layer_out_array, prev_layer_out);
    Py_RETURN_NONE;
    
fail:
    PyArray_Free(current_layer_delta_array, current_layer_delta);
    PyArray_Free(prev_layer_out_array, prev_layer_out);
    return NULL;
}


PyObject* conv_prev_layer_delta(PyObject* self, PyObject* args) {
    PyObject* current_layer_delta_array;
    PyObject* w;
    PyObject* out_array;

    /*  parse numpy array argument and value */
    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &current_layer_delta_array,
                                          &PyList_Type, &w,  /* list of numpy arrays */
                                          &PyArray_Type, &out_array
                                          )) {
        return NULL;
    }
    
    npy_intp current_layer_delta_shape[] = {0, 0, 0};
    double*** current_layer_delta = NULL;
    if (c_array_from_pyarray3d(&current_layer_delta_array, &current_layer_delta, (npy_intp*) current_layer_delta_shape) < 0) {
        goto fail;
    }
    
    npy_intp out_shape[] = {0, 0, 0};
    double*** out = NULL;
    if (c_array_from_pyarray3d(&out_array, &out, (npy_intp*) out_shape) < 0) {
        goto fail;
    }
    
    PyObject* obj_w_f = NULL;
    double*** w_f = NULL;
    npy_intp w_f_shape[] = {0, 0, 0};
    
    int f, x, y,
        z, i, j;
    for (f = 0; f < current_layer_delta_shape[2]; ++f) {
        obj_w_f = PyList_GetItem(w, f);
        
        if (c_array_from_pyarray3d(&obj_w_f, &w_f, (npy_intp*) w_f_shape) < 0) {
            goto fail;
        }
        
        for (x = 0; x < out_shape[0]; ++x) {
            for (y = 0; y < out_shape[1]; ++y) {
                for (z = 0; z < out_shape[2]; ++z) {
                    double conv = 0.0;
                    
                    for (i = 0; i < current_layer_delta_shape[0]; ++i) {
                        if ((x - i) >= 0 && (x - i) < w_f_shape[0]) {
                        
                            for (j = 0; j < current_layer_delta_shape[1]; ++j) {
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
        
        PyArray_Free(obj_w_f, w_f);
        w_f = NULL;
    }
    PyArray_Free(current_layer_delta_array, current_layer_delta);
    PyArray_Free(out_array, out);
    Py_RETURN_NONE;
    
fail:
    PyArray_Free(current_layer_delta_array, current_layer_delta);
    PyArray_Free(out_array, out);
    return NULL;
}
