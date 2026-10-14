# Experiment 23: Bayesian Network Construction

import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD

model = DiscreteBayesianNetwork([
    ("Smoking", "LungDisease"),
    ("Pollution", "LungDisease"),
    ("LungDisease", "Cough"),
    ("LungDisease", "Xray"),
    ("LungDisease", "Fatigue")
])

cpds = [
    TabularCPD("Smoking", 2, [[0.7], [0.3]]),
    TabularCPD("Pollution", 2, [[0.8], [0.2]]),
    TabularCPD("LungDisease", 2,
               [[0.99, 0.9, 0.9, 0.2],
                [0.01, 0.1, 0.1, 0.8]],
               evidence=["Smoking", "Pollution"],
               evidence_card=[2, 2]),
    TabularCPD("Cough", 2, [[0.8, 0.2], [0.2, 0.8]],
               evidence=["LungDisease"], evidence_card=[2]),
    TabularCPD("Xray", 2, [[0.95, 0.1], [0.05, 0.9]],
               evidence=["LungDisease"], evidence_card=[2]),
    TabularCPD("Fatigue", 2, [[0.7, 0.3], [0.3, 0.7]],
               evidence=["LungDisease"], evidence_card=[2])
]

model.add_cpds(*cpds)
print("Model Valid:", model.check_model())
print("Independencies:", model.get_independencies())
print("Active Trail:", model.active_trail_nodes("Smoking"))

G = nx.DiGraph(model.edges())
pos = nx.spring_layout(G, seed=10)
nx.draw(G, pos, with_labels=True, node_color="lightblue",
        node_size=2500, arrows=True)
plt.title("Medical Bayesian Network")
plt.show()
