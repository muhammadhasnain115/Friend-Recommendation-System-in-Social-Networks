import networkx as nx
import os
import pandas as pd
import matplotlib.pyplot as plt
import json
from joblib import Parallel, delayed
import time


file_path = r'C:\Users\eleni\Downloads\facebook_combined.txt'

print("File exists:", os.path.exists(file_path))
# Function to load the Facebook graph and convert to an undirected graph
def load_graph(file_path):
    G = nx.Graph()
    with open(file_path, 'r') as f:
        for line in f:
            node1, node2 = map(int, line.split())
            G.add_edge(node1, node2)
            G.add_edge(node2, node1)  # Add missing reciprocal edges
    return G

# Load the graph using the correct file path
graph = load_graph(file_path)
# Recommendation Algorithms


# Common Neighbors
# This function recommends friends based on the number of mutual friends (common neighbors). 
# The more mutual friends two users have, the higher their recommendation score.
def recommend_common_neighbors(G, node, top_n=5):
    neighbors = set(G.neighbors(node))
    scores = {}
    for neighbor in neighbors:
        for second_degree in G.neighbors(neighbor):
            if second_degree != node and second_degree not in neighbors:
                scores[second_degree] = scores.get(second_degree, 0) + 1
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]


# Jaccard Coefficient
# This function recommends friends based on the Jaccard coefficient,
#which measures similarity between two sets by dividing the size of their intersection by the size of their union.
# It is useful for determining the likelihood of forming a link based on shared connections.
def recommend_jaccard(G, node, top_n=5):
    scores = nx.jaccard_coefficient(G, [(node, other) for other in G.nodes() if other != node and not G.has_edge(node, other)])
    return sorted(scores, key=lambda x: x[2], reverse=True)[:top_n]


# Preferential Attachment
# This function recommends friends based on the principle of preferential attachment, 
#where nodes with higher degrees (more connections) are more likely to form new links.
# The score is based on the product of the degrees of the two nodes.
def recommend_preferential_attachment(G, node, top_n=5):
    scores = nx.preferential_attachment(G, [(node, other) for other in G.nodes() if other != node and not G.has_edge(node, other)])
    return sorted(scores, key=lambda x: x[2], reverse=True)[:top_n]

#Batch Processing for Recommendations
#Objective: Utilize parallel processing 
#to compute recommendations for potentially large subsets of the network efficiently.

# Batch Recommendations
def recommend_all(G, method='common_neighbors', top_n=5):
    recommend_fn = {
        'common_neighbors': recommend_common_neighbors,
        'jaccard': recommend_jaccard,
        'preferential_attachment': recommend_preferential_attachment
    }[method]
    recommendations = {node: recommend_fn(G, node, top_n) for node in G.nodes()}
    return recommendations

# Parallel Recommendations Function
def recommend_all_parallel(G, method='jaccard', top_n=5, node_limit=None, n_jobs=-1):
    recommend_fn = {
        'common_neighbors': recommend_common_neighbors,
        'jaccard': recommend_jaccard,
        'preferential_attachment': recommend_preferential_attachment
    }[method]

    if node_limit:
        nodes = list(G.nodes())[:node_limit]
    else:
        nodes = list(G.nodes())

    def recommend_fn_wrapper(node):
        return node, recommend_fn(G, node, top_n)

    results = Parallel(n_jobs=n_jobs)(delayed(recommend_fn_wrapper)(node) for node in nodes)
    return dict(results)

# Example Recommendations for node 0 using the Facebook dataset
print("Recommendations (Common Neighbors) for node 0:", recommend_common_neighbors(graph, 0))
print("Recommendations (Jaccard) for node 0:", recommend_jaccard(graph, 0))
print("Recommendations (Preferential Attachment) for node 0:", recommend_preferential_attachment(graph, 0))

# Time the execution
start = time.time()
all_recommendations = recommend_all_parallel(graph, method='jaccard', top_n=5, node_limit=1000, n_jobs=-1)
end = time.time()

print(f"Execution Time: {end - start:.2f} seconds")

# Save recommendations to file
with open('all_recommendations.json', 'w') as f:
    json.dump(all_recommendations, f, indent=2)

# Create a sample graph for visualization
sample_edges = list(graph.edges())[:50]
subgraph = nx.Graph()
subgraph.add_edges_from(sample_edges)

# Visualizing the sample graph
pos = nx.spring_layout(subgraph)
nx.draw(subgraph, pos, node_color='#A0CBE2', edge_color='#00bb5e', width=1, edge_cmap=plt.cm.Blues, with_labels=True)
plt.savefig("graph_sample.pdf")

# Print subgraph information using the summary method
print(subgraph.__str__())

num_nodes = subgraph.number_of_nodes()
num_edges = subgraph.number_of_edges()
avg_degree = sum(dict(subgraph.degree()).values()) / num_nodes

print(f"Graph Information\nNumber of Nodes: {num_nodes}\nNumber of Edges: {num_edges}\nAverage Degree: {avg_degree:.2f}")
