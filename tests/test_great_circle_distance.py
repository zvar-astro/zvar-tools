from zvartools.spatial import great_circle_distance

import numpy as np
import time

# generate 12k random points on the sphere
n = 12000
ra1 = np.random.uniform(0, 360, n)
dec1 = np.random.uniform(-90, 90, n)
ra2 = np.random.uniform(0, 360, n)
dec2 = np.random.uniform(-90, 90, n)

# time how long it takes to calculate the distances
start = time.time()
for i in range(n):
    great_circle_distance(ra1[i], dec1[i], ra2[i], dec2[i])
end = time.time()

print(f"Great circle distance calculation for {n} pairs took {end - start:.2f} seconds")
