import convnetlib
import numpy as np

filters_count = 2
s = 1
filter_size = 5
X = np.zeros((28, 28, 1))
w = [np.zeros((filter_size, filter_size, 1)) for _ in xrange(filters_count)]
b = [0] * filters_count

out = np.zeros((24, 24, filters_count))
res = convnetlib.conv_forward(X, w, b, s, out)
# print(convnetlib.test(out))
