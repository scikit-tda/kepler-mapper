"""
nearest_nodes example based on breast cancer data.
"""

from plot_breast_cancer import *
from sklearn import neighbors, preprocessing

# new patient data incoming
i = np.random.randint(len(X))
new_patient_data = 1.05*X[i]
new_patient_data = new_patient_data.reshape(1, -1)

# re-use lens1 model
newlens1 = model.decision_function(new_patient_data)

# re-construct lens2 model
X_norm = np.linalg.norm(X, axis=1)
scaler = preprocessing.MinMaxScaler()
scaler.fit(X_norm.reshape(-1, 1))

newlens2 = scaler.transform(np.linalg.norm(new_patient_data, axis=1).reshape(1, -1))

newlens = np.c_[newlens1, newlens2]

# find nearest nodes
nn = neighbors.NearestNeighbors(n_neighbors=3)
node_ids = mapper.nearest_nodes(newlens, new_patient_data, graph, mapper.cover, lens, X, nn)

print("Nearest nodes:")
for node_id in node_ids:
    diags = y[graph['nodes'][node_id]]
    print("  {}: diagnosis {:.1f}%".format(node_id, np.sum(diags)*100.0/len(diags)))
