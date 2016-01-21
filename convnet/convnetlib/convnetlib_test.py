import convnetlib
import numpy as np

filters_count = 2
s = 1
filter_size = 5
X = np.zeros((28, 28, 1))
w = [np.zeros((filter_size, filter_size, 1)) for _ in xrange(filters_count)]
b = [0] * filters_count

out = np.zeros((24, 24, filters_count))
out2 = np.zeros((24, 24, filters_count))

pool_shape = (12, 12, filters_count)
max_indices = np.zeros(pool_shape, dtype=(int, 2))
pool_out = np.zeros(pool_shape)

for _ in xrange(10000000):
    # convnetlib.conv_forward(X, w, b, s, out)
    # convnetlib.conv_backward(out2, X, w, b)
    # convnetlib.conv_prev_layer_delta(out, w, X)
    # convnetlib.pool_forward(out, s, 2, max_indices, pool_out)
    res = np.zeros(out.shape)
    convnetlib.pool_prev_layer_delta(pool_out, max_indices, res)

# print(convnetlib.test(out))
