#%%
import matplotlib.pyplot as plt
import h5py
import numpy as np

#%%
def loadmat_h5py(file):
    output = {}
    with h5py.File(file, 'r') as f:
        for k, v in f.items():
            output[k] = np.array(v)
    return output

#%%
f = h5py.File("diffuser_10s.mat", "r")
diffuser_arr = np.array(f["diffuser"]).squeeze()
diffuser_captures = []
for i in range(len(diffuser_arr)):
    diffuser_captures.append(np.array(f[diffuser_arr[i]]).squeeze())
diffuser_captures = np.array(diffuser_captures)

#%%
f = h5py.File("scanned_10s_512res.mat", "r")
scanned_arr = np.array(f["scanned"]).squeeze()
scanned_captures = []
for i in range(len(scanned_arr)):
    scanned_captures.append(np.array(f[scanned_arr[i]]).squeeze())
scanned_captures = np.array(scanned_captures)
scanned_img = np.sum(scanned_captures[0, :, :], axis=2)

#%%
# Each entry of diffuser_captures is a 1s exposure
diffuser_sum = np.sum(diffuser_captures, axis=0)
diffuser = diffuser_sum[0::2] + diffuser_sum[1::2]
diffuser = diffuser[0::2] + diffuser[1::2]

scanned = np.sum(scanned_captures[0, :, :], axis=(0, 1))  # 10 s exposure

# Shift
diff_peak = np.argmax(diffuser)
scan_peak = np.argmax(scanned)
offset = diff_peak - scan_peak
diffuser = np.roll(diffuser, -offset)

# Focus
n_min = 500
n_max = 1200
diffuser = diffuser[n_min:n_max]
# diffuser = diffuser/np.max(diffuser)
diffuser = diffuser/np.sum(diffuser)
scanned = scanned[n_min:n_max]
# scanned = scanned/np.max(scanned)
scanned = scanned/np.sum(scanned)

t = np.linspace(n_min, n_max-1, n_max-n_min) * 16e-12 * 1e9;
plt.plot(t, diffuser, label="diffused")
plt.plot(t, scanned, label="scanned")
plt.legend()
plt.xlabel("time (ns)")
plt.yscale("log")
plt.savefig("plot.pdf")
plt.show()


#%%
def remove_dc_from_spad_edge(spad, ambient, grad_th=1e3, n_std=1.):
    """
    Create a "bounding box" that is bounded on the left and the right
    by using the gradients and below by using the ambient estimate.
    """
    # Detect edges:
    assert len(spad.shape) == 1
    edges = np.abs(np.diff(spad)) > grad_th
    print(np.argwhere(edges))
    first = np.nonzero(edges)[0][1] + 1  # Want the right side of the first edge
    last = np.nonzero(edges)[0][-1]      # Want the left side of the second edge
    below = ambient + n_std*np.sqrt(ambient)
    # Walk first and last backward and forward until we encounter a value below the threshold
    while first >= 0 and spad[first] > below:
        first -= 1
    while last < len(spad) and spad[last] > below:
        last += 1
    spad[:first] = 0.
    spad[last+1:] = 0.
    print(first)
    print(last)
    return np.clip(spad - ambient, a_min=0., a_max=None)

# Try removing ambient
diffuser_sum = np.sum(diffuser_captures, axis=0)
diffuser = diffuser_sum[0::2] + diffuser_sum[1::2]
diffuser = diffuser[0::2] + diffuser[1::2]
diffuser_clip = diffuser[500:2048].astype('float')
scanned = np.sum(scanned_captures[0, :, :], axis=(0, 1))  # 10 s exposure
scanned_clip = scanned[400:2048]
diff_amb = np.mean(diffuser_clip[:100])
scan_amb = np.mean(scanned_clip[:100])
diffuser_no_amb = remove_dc_from_spad_edge(diffuser_clip,
                                           diff_amb,
                                           grad_th=5*np.sqrt(2*diff_amb))
scanned_no_amb = remove_dc_from_spad_edge(scanned_clip,
                                          scan_amb,
                                          grad_th=5*np.sqrt(2*scan_amb))
diff_peak = np.argmax(diffuser_no_amb)
scan_peak = np.argmax(scanned_no_amb)
offset = diff_peak - scan_peak
diffuser_no_amb = np.roll(diffuser_no_amb, -offset)

plt.plot(diffuser_no_amb, label="diffuser_no_amb")
plt.plot(scanned_no_amb, label="scanned_no_amb")
plt.yscale("log")
plt.legend()
plt.show()

