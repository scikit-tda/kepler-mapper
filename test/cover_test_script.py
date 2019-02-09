import numpy as np
from kmapper.cover import Cover

# uniform data:
data = np.arange(0, 1000).reshape((1000, 1))
lens = data
cov = Cover(10, 0.5, verbose=0)


def overlap(c1, c2):
    ints = set(c1).intersection(set(c2))
    return len(ints) / max(len(c1), len(c2))


# Prefix'ing the data with an ID column
ids = np.array([x for x in range(lens.shape[0])])
lens = np.c_[ids, lens]


bins = cov.fit(lens)
cube_entries = cov.transform(lens, bins)

for i, hypercube in enumerate(cube_entries):
    print(
        "There are %s points in cube %s/%s" % (hypercube.shape[0], i, len(cube_entries))
    )

print()
for i, (c1, c2) in enumerate(zip(cube_entries, cube_entries[1:])):
    print("Overlap %s" % (overlap(c1[:, 0], c2[:, 0])))
