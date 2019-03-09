import numpy as np


class SID:
    def __init__(self, sid_bins, min_val, max_val, offset):
        self.sid_bins = sid_bins
        self.min_val = min_val
        self.max_val = max_val
        self.offset = offset

        # Derived quantities
        self.alpha = min_val + offset
        self.beta = max_val + offset
        bin_edges = np.array(range(sid_bins + 1)).astype(np.float32)
        self.sid_bin_edges = np.array(np.exp(np.log(self.alpha) +
                                             bin_edges / self.sid_bins * np.log(self.beta / self.alpha)))
        self.sid_bin_values = (self.sid_bin_edges[:-1] + self.sid_bin_edges[1:]) / 2 - self.offset
        self.sid_bin_values = np.append(self.sid_bin_values, [self.max_val, self.min_val])
        # Do the above so that:
        # self.sid_bin_values[-1] = self.min_val < self.sid_bin_values[0]
        # and
        # self.sid_bin_values[sid_bins] = self.max_val > self.sid_bin_values[sid_bins-1]

    def get_sid_index_from_value(self, arr):
        """
        Given an array of values in the range [min_val, max_val], return the
        indices of the bins they correspond to
        :param arr: The array to turn into indices.
        :return: The array of indices.
        """
        sid_index = np.floor(self.sid_bins * (np.log(arr + self.offset) - np.log(self.alpha)) /
                                             (np.log(self.beta) - np.log(self.alpha))).astype(np.int32)
        sid_index = np.clip(sid_index, -1, self.sid_bins)
        # An index of -1 indicates min_val, while self.sid_bins indicates max_val
        return sid_index

    def get_value_from_sid_index(self, sid_index):
        """
        Given an array of indices in the range {-1, 0,...,sid_bins},
        return the representative value of the selected bin.
        :param sid_index: The array of indices.
        :return: The array of values correspondding to those indices
        """
        return np.take(self.sid_bin_values, sid_index)


class AddSIDDepth:
    """Creates a copy of the depth image where the depth value has been replaced
    by the SID-discretized index

    Discretizes depth into |sid_bins| number of bins, where the edges of the bins are
    given by

    t_i = exp(log(alpha) + i/K*log(beta/alpha))

    for i in {0,...,K}.

    Works in numpy.

    offset is a shift term that we subtract from alpha
    """

    def __init__(self, sid_bins, max_val, min_val, offset, key):
        """
        :param sid_obj: The SID object to use to convert between indices and depth values and vice versa
        :param key: The key (in sample) of the depth map to use.
        """
        self.key = key  # Key of the depth image to convert to SID form.
        self.sid_obj = SID(sid_bins, max_val, min_val, offset)

    def __call__(self, sample):
        """Computes an array with indices, and also an array with
        0's and 1's that makes computing the ordinal regression loss easier later.

        Index array gives the per-pixel bin index of the depth value.
        0's and 1's array has a vector of length |sid_bins| for each pixel that is
        1.0 up to (but not including) the index of the depth value, and 0.0 for the rest.
        Example:
             If depth_sid_index assigns some pixel to be bin 4 (out of 7 bins), then the
             vector for the same pixel in depth_sid is
              0 1 2 3 4 5 6
             [1 1 1 1 0 0 0]
             Note: The most 1's possible is n, where n is the number of bins:
             [1 1 1 1 1 1 1]
             The fewest is 0.
        """
        depth = sample[self.key]
        sample[self.key + "_sid_index"] = self.sid_obj.get_sid_index_from_value(depth)
        K = np.zeros((self.sid_obj.sid_bins,) + depth.shape)
        print(K.shape)
        for i in range(self.sid_obj.sid_bins):  # i = {0, ..., self.sid_bins - 1}
            K[i, ...] = K[i, ...] + i * np.ones(depth.shape)
        sample[self.key + "_sid"] = (K < sample[self.key + "_sid_index"][np.newaxis, ...]).astype(np.int32)
        return sample


if __name__ == '__main__':
    K = 68
    bin_edges = np.array(range(K + 1)).astype(np.float32)
    dorn_decode = np.exp((bin_edges - 1) / 25 - 0.36)
    d0 = dorn_decode[0]
    d1 = dorn_decode[1]
    alpha = (2 * d0 ** 2) / (d1 + d0)
    print(alpha)
    beta = alpha * np.exp(K * np.log(2 * d0 / alpha - 1))
    print(beta)

    sid_nyuv2_dorn = SID(K, alpha, beta, 0.)
    print(sid_nyuv2_dorn.sid_bin_edges)
    print(sid_nyuv2_dorn.sid_bin_values)
    arr = np.array([0., 0.4, 2, 9, 10])

    sid_index = sid_nyuv2_dorn.get_sid_index_from_value(arr)
    print(sid_index)
    values = sid_nyuv2_dorn.get_value_from_sid_index(sid_index)
    print(values)

    ###

    transform = AddSIDDepth(K, alpha, beta, 0., "test")
    sample = {"test": np.array([[0, 0.4, 2, 9, 10]])}
    output = transform(sample)
    print(output)

