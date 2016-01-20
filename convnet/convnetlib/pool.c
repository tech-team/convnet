#include "pool.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL convnetlib_ARRAY_API
#include <numpy/arrayobject.h>
#include <stdio.h>

#include "util.h"

PyObject* pool_forward(PyObject* self, PyObject* args) {
    PyObject* data_array;
    int stride;
    int filter_size;
    PyObject* max_indices_array;
    PyObject* out_array;

    /*  parse numpy array argument and value */
    if (!PyArg_ParseTuple(args, "O!iiO!O!", &PyArray_Type, &data_array,
                                            &stride,
                                            &filter_size,
                                            &PyArray_Type, &max_indices_array,
                                            &PyArray_Type, &out_array
                                            )) {
        return NULL;
    }
    
    npy_intp data_shape[] = {0, 0, 0};
    double*** data = NULL;
    if (c_array_from_pyarray3d(&data_array, &data, (npy_intp*) data_shape) < 0) {
        goto fail;
    }
    
    npy_intp out_shape[] = {0, 0, 0};
    double*** out = NULL;
    if (c_array_from_pyarray3d(&out_array, &out, (npy_intp*) out_shape) < 0) {
        goto fail;
    }
    
    if (out_shape[2] != data_shape[2]){
        PyErr_SetString(PyExc_ValueError, "data_array's and out_array's shape[2] have to be the same");
        goto fail;
    }
    
    int f = filter_size;
    int k0 = 0,
        k1 = 0;
        
    for (int x = 0; x < out_shape[0]; ++x) {
        int x_offset = x * stride;
        k1 = 0;
        for (int y = 0; y < out_shape[1]; ++y) {
            int y_offset = y * stride;
            for (int z = 0; z < out_shape[2]; ++z) {
                
                
                /* searching max_index */
                int max_i = x_offset,
                    max_j = y_offset;
                double max_value = -1;
                if (max_i < data_shape[0] && max_i < x_offset + f &&
                    max_j < data_shape[1] && max_j < y_offset + f) {
                    
                    max_value = data[max_i][max_j][z];
                    for (int i = x_offset; i < data_shape[0] && i < x_offset + f; ++i) {
                        for (int j = y_offset; j < data_shape[1] && j < y_offset + f; ++j) {
                            double value = data[i][j][z];
                            if (value > max_value) {
                                max_i = i;
                                max_j = j;
                                max_value = value;
                            }
                        }
                    }
                    
                    out[k0][k1][z] = max_value;
                    
                    void* item_ptr_0 = PyArray_GETPTR4((PyArrayObject*) max_indices_array, k0, k1, z, 0);
                    void* item_ptr_1 = PyArray_GETPTR4((PyArrayObject*) max_indices_array, k0, k1, z, 1);
                    
                    if (PyArray_SETITEM((PyArrayObject*) max_indices_array, (char*) item_ptr_0, PyInt_FromLong((long) max_i)) < 0) {
                        goto fail;
                    }
                    if (PyArray_SETITEM((PyArrayObject*) max_indices_array, (char*) item_ptr_1, PyInt_FromLong((long) max_j)) < 0) {
                        goto fail;
                    }
                }
            }
            ++k1;
        }
        ++k0;
    }
    PyArray_Free(data_array, data);
    PyArray_Free(out_array, out);
    Py_RETURN_NONE;
    
fail:
    PyArray_Free(data_array, data);
    PyArray_Free(out_array, out);
    return NULL;
}


PyObject* pool_prev_layer_delta(PyObject* self, PyObject* args) {
    PyObject* current_layer_delta_array;
    PyObject* max_indices_array;
    PyObject* out_array;

    /*  parse numpy array argument and value */
    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &current_layer_delta_array,
                                          &PyArray_Type, &max_indices_array,
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
    
    for (int x = 0; x < current_layer_delta_shape[0]; ++x) {
        for (int y = 0; y < current_layer_delta_shape[1]; ++y) {
            for (int z = 0; z < current_layer_delta_shape[2]; ++z) {
                
                void* item_ptr_0 = PyArray_GETPTR4((PyArrayObject*) max_indices_array, x, y, z, 0);
                void* item_ptr_1 = PyArray_GETPTR4((PyArrayObject*) max_indices_array, x, y, z, 1);
                
                PyObject* max_i_obj = PyArray_GETITEM((PyArrayObject*) max_indices_array, (char*) item_ptr_0);
                PyObject* max_j_obj = PyArray_GETITEM((PyArrayObject*) max_indices_array, (char*) item_ptr_1);
                
                long max_i = PyInt_AsLong(max_i_obj);
                long max_j = PyInt_AsLong(max_j_obj);
                
                out[max_i][max_j][z] = current_layer_delta[x][y][z];
            }
        }
    }
    
    PyArray_Free(current_layer_delta_array, current_layer_delta);
    PyArray_Free(out_array, out);
    Py_RETURN_NONE;
    
fail:
    PyArray_Free(current_layer_delta_array, current_layer_delta);
    PyArray_Free(out_array, out);
    return NULL;
}
