import numpy as np

d1 = np.loadtxt("descriptors/descriptor_s034.txt", delimiter = ' ')

d2 = np.loadtxt("descriptors/descriptor_r034.txt", delimiter = ' ')

thr_distance = 0.1

for i in range(0, len(d1)):
    for j in range(0, len(d2)):
        if np.linalg.norm(d1[i] - d2[j]) < thr_distance:
            print(f"Descriptor match! Synthetic {i+1} and RGB {j+1}")
        