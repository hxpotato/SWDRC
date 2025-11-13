import networkx as nx
import pandas as pd
import numpy as np
import os
path_folder = "H:/WindowStr/WinDtwPath/Path_cal/P005/20_8/"
nwork_save_folder = "H:/WindowStr/WinDtwPath/Path_cal/P005/20_8_network/"
def create_network_from_matrix(matrix,matrix_size):
    G_directed = nx.DiGraph()
    G_undirected = nx.Graph()
    matrix = np.array(matrix)
    G_directed.add_nodes_from(range(matrix_size))
    G_undirected.add_nodes_from(range(matrix_size))
    for i in range(matrix_size):
        for j in range(i + 1, matrix_size):
            weight = matrix[i][j]
            if weight > 0:
                G_directed.add_edge(i, j, weight=abs(weight))
                G_undirected.add_edge(i, j, weight=abs(weight))
            elif weight < 0:
                G_directed.add_edge(j, i, weight=abs(weight))
                G_undirected.add_edge(i, j, weight=abs(weight))
    return G_directed, G_undirected
def calculate_network_metrics(G):
    metrics = {}
    is_directed = G.is_directed()
    G_undirected = G.to_undirected() if is_directed else G
    is_connected_func = nx.is_weakly_connected if is_directed else nx.is_connected
    if is_connected_func(G):
        metrics['Average Path Length'] = nx.average_shortest_path_length(G_undirected)
        metrics['Diameter'] = nx.diameter(G_undirected)
    else:
        metrics['Average Path Length'] = float('inf')
        metrics['Diameter'] = float('inf')
    metrics['Clustering Coefficient'] = nx.average_clustering(G)
    metrics['Degree Centrality'] = np.mean(list(nx.degree_centrality(G).values()))
    metrics['Closeness Centrality'] = np.mean(list(nx.closeness_centrality(G).values()))
    metrics['Betweenness Centrality'] = np.mean(list(nx.betweenness_centrality(G).values()))
    metrics['Density'] = nx.density(G)
    metrics['Assortativity'] = nx.degree_assortativity_coefficient(G)
    metrics['Global Efficiency'] = nx.global_efficiency(G_undirected)
    metrics['Local Efficiency'] = nx.local_efficiency(G_undirected)
    return metrics
def analyze_networks(graphs, file_names):
    results = []
    for idx, G in enumerate(graphs):
        result = calculate_network_metrics(G)
        result['File Name'] = file_names[idx]  
        results.append(result)
    df = pd.DataFrame(results)
    df.to_csv('network_analysis_results.csv', index=False)
    return df
def analyze_networks_node(graphs, file_names):
    results = []
    for idx, G in enumerate(graphs):
        result = compute_network_metrics_node(G)
        result['File Name'] = file_names[idx]  
        results.append(result)
    df = pd.DataFrame(results)
    return df
def calculate_network_impl(G):

    if G.is_directed():
        return nx.is_weakly_connected(G)  
    else:
        return nx.is_connected(G)
def compute_network_metrics_node(G):
    metrics = {}
    if nx.is_directed(G):
        connected = nx.is_strongly_connected(G)
    else:
        connected = nx.is_connected(G)
    for node in G.nodes():
        metrics[node] = {}
        metrics[node]['Degree'] = G.degree(node)
        if nx.is_weighted(G):
            metrics[node]['Weighted Degree'] = sum(weight for _, _, weight in G.edges(node, data='weight'))
        else:
            metrics[node]['Weighted Degree'] = 'Graph is unweighted'
    clustering = nx.clustering(G)
    for node in G.nodes():
        metrics[node]['Clustering Coefficient'] = clustering[node]
    if connected:
        closeness_centrality = nx.closeness_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06)
    else:
        closeness_centrality = nx.closeness_centrality(G, wf_improved=False)
        betweenness_centrality = nx.betweenness_centrality(G, endpoints=True)
        eigenvector_centrality = {}
        for component in nx.strongly_connected_components(G) if nx.is_directed(G) else nx.connected_components(G):
            subgraph = G.subgraph(component)
            try:
                ec = nx.eigenvector_centrality(subgraph, max_iter=1000, tol=1e-06)
                eigenvector_centrality.update(ec)
            except nx.PowerIterationFailedConvergence as e:
                for n in subgraph.nodes():
                    eigenvector_centrality[n] = 0
    for node in G.nodes():
        metrics[node]['Closeness Centrality'] = closeness_centrality.get(node, float('inf'))  
        metrics[node]['Betweenness Centrality'] = betweenness_centrality.get(node, 0)
        metrics[node]['Eigenvector Centrality'] = eigenvector_centrality.get(node, 0)
    pagerank = nx.pagerank(G)
    for node in G.nodes():
        metrics[node]['PageRank'] = pagerank[node]
    return metrics
def load_mats(folder,patient_type,delimiter):
    class_matrices=[]
    files_names = []
    for file_name in os.listdir(folder):
        if file_name.endswith('.txt') and patient_type in file_name:
            file_path = os.path.join(folder, file_name)
            matrix = np.loadtxt(file_path,delimiter=delimiter)
            class_matrices.append(matrix)
            files_names.append(file_name)
    class_matrices = np.array(class_matrices)
    return class_matrices, files_names
def main():
    window_size = [i for i in range(10, 55, 5)]
    stride = [i for i in range(1, 11)]
    mat_size = 90
    for w in window_size:
        for s in stride:
            now_path_folder = path_folder + f"{w}_{s}/"
            print("正在读取文件夹"+now_path_folder)
            CN_mats,CN_files_name = load_mats(now_path_folder,'CN',delimiter=' ')
            AD_mats,AD_files_name = load_mats(now_path_folder,'AD',delimiter=' ')
            CN_network_directed = []
            CN_network_undirected = []
            AD_network_directed = []
            AD_network_undirected = []
            for CN_mat in CN_mats:
                G_directed,G_undirected = create_network_from_matrix(CN_mat,mat_size)
                CN_network_directed.append(G_directed)
                CN_network_undirected.append(G_undirected)
            for AD_mat in AD_mats:
                G_directed,G_undirected = create_network_from_matrix(AD_mat,mat_size)
                AD_network_directed.append(G_directed)
                AD_network_undirected.append(G_undirected)
            CN_results_Dir = analyze_networks(CN_network_directed, CN_files_name)
            AD_results_Dir = analyze_networks(AD_network_directed, AD_files_name)
            CN_results_Dir.to_csv(nwork_save_folder + f"{w}_{s}_CN_directed_results.csv", index=False)
            AD_results_Dir.to_csv(nwork_save_folder + f"{w}_{s}_AD_directed_results.csv", index=False)
            CN_results_UnDir = analyze_networks(CN_network_undirected, CN_files_name)
            AD_results_UnDir = analyze_networks(AD_network_undirected, AD_files_name)
            CN_results_UnDir.to_csv(nwork_save_folder + f"{w}_{s}_CN_undirected_results.csv", index=False)
            AD_results_UnDir.to_csv(nwork_save_folder + f"{w}_{s}_AD_undirected_results.csv", index=False)
            CN_results_Dir_node = analyze_networks_node(CN_network_directed, CN_files_name)
            AD_results_Dir_node = analyze_networks_node(AD_network_directed, AD_files_name)
            CN_results_Dir_node.to_csv(nwork_save_folder + f"{w}_{s}_CN_directed_results_node.csv", index=False)
            AD_results_Dir_node.to_csv(nwork_save_folder + f"{w}_{s}_AD_directed_results_node.csv", index=False)
            CN_results_UnDir_node = analyze_networks_node(CN_network_undirected, CN_files_name)
            AD_results_UnDir_node = analyze_networks_node(AD_network_undirected, AD_files_name)
            CN_results_UnDir_node.to_csv(nwork_save_folder + f"{w}_{s}_CN_undirected_results_node.csv", index=False)
            AD_results_UnDir_node.to_csv(nwork_save_folder + f"{w}_{s}_AD_undirected_results_node.csv", index=False)
def main2():
    mat_size = 90
    now_path_folder = path_folder
    print("正在读取文件夹" + now_path_folder)
    CN_mats, CN_files_name = load_mats(now_path_folder, 'CN', delimiter=' ')
    AD_mats, AD_files_name = load_mats(now_path_folder, 'AD', delimiter=' ')
    CN_network_directed = []
    CN_network_undirected = []
    AD_network_directed = []
    AD_network_undirected = []
    for CN_mat in CN_mats:
        G_directed, G_undirected = create_network_from_matrix(CN_mat, mat_size)
        CN_network_directed.append(G_directed)
        CN_network_undirected.append(G_undirected)
    for AD_mat in AD_mats:
        G_directed, G_undirected = create_network_from_matrix(AD_mat, mat_size)
        AD_network_directed.append(G_directed)
        AD_network_undirected.append(G_undirected)
    CN_results_Dir = analyze_networks(CN_network_directed, CN_files_name)
    AD_results_Dir = analyze_networks(AD_network_directed, AD_files_name)
    CN_results_Dir.to_csv(nwork_save_folder + f"CN_directed_results.csv", index=False)
    AD_results_Dir.to_csv(nwork_save_folder + f"AD_directed_results.csv", index=False)
    CN_results_UnDir = analyze_networks(CN_network_undirected, CN_files_name)
    AD_results_UnDir = analyze_networks(AD_network_undirected, AD_files_name)
    CN_results_UnDir.to_csv(nwork_save_folder + f"CN_undirected_results.csv", index=False)
    AD_results_UnDir.to_csv(nwork_save_folder + f"AD_undirected_results.csv", index=False)
    CN_results_Dir_node = analyze_networks_node(CN_network_directed, CN_files_name)
    AD_results_Dir_node = analyze_networks_node(AD_network_directed, AD_files_name)
    CN_results_Dir_node.to_csv(nwork_save_folder + f"CN_directed_results_node.csv", index=False)
    AD_results_Dir_node.to_csv(nwork_save_folder + f"AD_directed_results_node.csv", index=False)
    CN_results_UnDir_node = analyze_networks_node(CN_network_undirected, CN_files_name)
    AD_results_UnDir_node = analyze_networks_node(AD_network_undirected, AD_files_name)
    CN_results_UnDir_node.to_csv(nwork_save_folder + f"CN_undirected_results_node.csv", index=False)
    AD_results_UnDir_node.to_csv(nwork_save_folder + f"AD_undirected_results_node.csv", index=False)
if __name__ == '__main__':
    main2()
