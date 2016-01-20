#ifndef _CONVNETLIB_POOL_H_
#define _CONVNETLIB_POOL_H_

#include <Python.h>

PyObject* pool_forward(PyObject* self, PyObject* args);
PyObject* pool_prev_layer_delta(PyObject* self, PyObject* args);

#endif /* _CONVNETLIB_CONV_H_ */
