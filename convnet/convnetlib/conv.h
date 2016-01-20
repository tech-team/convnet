#ifndef _CONVNETLIB_CONV_H_
#define _CONVNETLIB_CONV_H_

#include <Python.h>

PyObject* conv_forward(PyObject* self, PyObject* args);
PyObject* conv_backward(PyObject* self, PyObject* args);
PyObject* conv_prev_layer_delta(PyObject* self, PyObject* args);

#endif /* _CONVNETLIB_CONV_H_ */
