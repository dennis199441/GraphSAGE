import networkx as nx
import numpy as np
import os, math, pickle, argparse, random

WALK_LEN=5
N_WALKS=50

def load_id_map(G):
	id_map = {}
	nodes = G.nodes()
	i = 0
	for node in nodes:
		id_map[node] = i
		i += 1
	return id_map

def load_class_map(G):
	class_map = {}
	nodes = G.nodes()
	pickle_in = open('C:/Users/cwxxcheun/Desktop/Other/github/cmsc5721-project/sandp500_data/sector_dict.pickle', "rb")
	sector_dict = pickle.load(pickle_in)

	sector_mapping = {}
	sector_mapping['Technology'] = 0
	sector_mapping['Financial Services'] = 1
	sector_mapping['Consumer Cyclical'] = 2
	sector_mapping['Utilities'] = 3
	sector_mapping['Communication Services'] = 4
	sector_mapping['Energy'] = 5
	sector_mapping['Industrials'] = 6
	sector_mapping['Real Estate'] = 7
	sector_mapping['Basic Materials'] = 8
	sector_mapping['Consumer Defensive'] = 9
	sector_mapping['Healthcare'] = 10

	for node in nodes:
		class_map[node] = [0] * 11
		class_map[node][sector_mapping[sector_dict[node]]] = 1

	return class_map

def test_val_annotation(G, test_size=0.1, val_size=0.1):
	nodes = G.nodes()
	test_num = int(len(nodes) * test_size)
	val_num = int(len(nodes) * val_size)

	test_set = []
	val_set = []
	for i in range(test_num):
		choice_node = random.choice(list(nodes))
		if choice_node not in val_set:
			test_set.append(choice_node)

	for i in range(val_num):
		choice_node = random.choice(list(nodes))
		if choice_node not in test_set:
			val_set.append(choice_node)

	test_attr = {}
	val_attr = {}
	for node in nodes:
		if node in test_set:
			test_attr[node] = True
		else:
			test_attr[node] = False

	for node in nodes:
		if node in val_set:
			val_attr[node] = True
		else:
			val_attr[node] = False

	for k, v in test_attr.items():
		G.nodes[k]['test'] = v

	for k, v in val_attr.items():
		G.nodes[k]['val'] = v

def load_feats(G):
	nodes = G.nodes()
	feat_list = []
	for node in nodes:
		feat = []
		feat.append(G.nodes[node]['price'])
		feat.append(G.nodes[node]['mean_return'])
		feat.append(G.nodes[node]['std_return'])
		feat_list.append(feat)

	return np.array(feat_list)

def custom_load_data(filename, walksname, normalize=True):
	# filename = '/Users/dennis199441/Documents/GitHub/cmsc5721-project/network_data/daily_net/metadata_stocknet_timescale_250threshold_0.6/stocknet_20180110_20190109.pickle'
	pickle_in = open(filename, "rb")
	G = pickle.load(pickle_in)
	if isinstance(list(G.nodes())[0], int):
		conversion = lambda n : int(n)
	else:
		conversion = lambda n : n

	id_map = load_id_map(G)
	class_map = load_class_map(G)
	test_val_annotation(G, test_size=0.1, val_size=0.1)
	feats = load_feats(G)
	walks = []

	## Remove all nodes that do not have val/test annotations
	## (necessary because of networkx weirdness with the Reddit data)
	broken_count = 0
	for node in G.nodes():
		if not 'val' in G.node[node] or not 'test' in G.node[node]:
			G.remove_node(node)
			broken_count += 1
	print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

	## Make sure the graph has edge train_removed annotations
	## (some datasets might already have this..)
	print("Loaded data.. now preprocessing..")
	for edge in G.edges():
		if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
			G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
			G[edge[0]][edge[1]]['train_removed'] = True
		else:
			G[edge[0]][edge[1]]['train_removed'] = False

	if normalize and not feats is None:
		from sklearn.preprocessing import StandardScaler
		train_ids = np.array([id_map[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
		train_feats = feats[train_ids]
		scaler = StandardScaler()
		scaler.fit(train_feats)
		feats = scaler.transform(feats)

	if walksname:
		with open(walksname) as fp:
			for line in fp:
				walks.append(map(conversion, line.split()))

	return G, feats, id_map, walks, class_map

def run_random_walks(G, nodes, num_walks=N_WALKS):
	pairs = []
	for count, node in enumerate(nodes):
		if G.degree(node) == 0:
			continue
		for i in range(num_walks):
			curr_node = node
			for j in range(WALK_LEN):
				next_node = random.choice(list(G.neighbors(curr_node)))
				# self co-occurrences are useless
				if curr_node != node:
					pairs.append((node,curr_node))
				curr_node = next_node
		if count % 1000 == 0:
			print("Done walks for", count, "nodes")
	return pairs

if __name__ == "__main__":
	path = '/Users/dennis199441/Documents/GitHub/cmsc5721-project/network_data/daily_net/metadata_stocknet_timescale_250threshold_0.6/'
	list_dir = os.listdir(path)

	for file in list_dir:
		if file.endswith('.pickle'):
			filename = path + file
			out_file = filename.replace('.pickle','-walks.txt')
			G, feats, id_map, walks, class_map = custom_load_data(filename=filename)
			nodes = [n for n in G.nodes() if not G.node[n]["val"] and not G.node[n]["test"]]
			G = G.subgraph(nodes)
			pairs = run_random_walks(G, nodes)
			with open(out_file, "w") as fp:
				fp.write("/n".join([str(p[0]) + "/t" + str(p[1]) for p in pairs]))
	

