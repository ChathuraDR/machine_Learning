import numpy as np
import matplotlib.pyplot as plt


greyhounds = 500
labs = 500

#
# more infomation about np.random.randn()
# https://www.geeksforgeeks.org/numpy-random-randn-python/
# you may also need python3-tk package. apt install python3-tk

grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labs)

plt.hist([grey_height, lab_height], stacked=True, color=['r','b'])	# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html
plt.show()