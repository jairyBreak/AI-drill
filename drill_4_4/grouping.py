import json
import networkx as nx

# read the topology
with open('p4app.json', 'r') as f:
    topo_data = json.load(f)['topology']

G = nx.DiGraph()

# build link and get the bandwidth
for link in topo_data['links']:
    u, v = link[0], link[1]
    attrs = link[2] if len(link) > 2 else {}
    bw = attrs.get('bw', 1.0) # else then bandwidth = 1 (host to leaf)
    G.add_edge(u, v, bw=bw)
    G.add_edge(v, u, bw=bw)

# find Leaf Switches (connect with host)
hosts = topo_data['hosts'].keys()
leaf_switches = set()
for h in hosts:
    for neighbor in G.neighbors(h):
        leaf_switches.add(neighbor)

print(f"Leaf Switches: {leaf_switches}")

# build Quiver with CF
Q = nx.DiGraph()
for u, v in G.edges():
    Q.add_edge(u, v, labels=set())

for src in leaf_switches:
    for dst in leaf_switches:
        if src == dst: continue
        try:
            for path in nx.all_shortest_paths(G, source=src, target=dst): #path:all links from src to dst 
                bottleneck_bw = float('inf')
                for i in range(len(path) - 1):
                    a = path[i]
                    b = path[i+1]
                    link_bw = G[a][b]['bw']
                    # CF (Capacity Factor)
                    if i == 0:
                        # leaf to switch (first hop)
                        cf = float('inf')
                    else:
                        # cf = capacity(src, a) / capacity(a, b)
                        cf = bottleneck_bw / link_bw
                    # find the min bandwidth
                    bottleneck_bw = min(bottleneck_bw, link_bw)

                    # "src->dst_CF"
                    label = f"{src}->{dst}_CF:{cf}"
                    Q[a][b]['labels'].add(label)
                    
        except nx.NetworkXNoPath:
            pass

# ==========================================
# 3. 找出對稱路徑群組，並計算總權重
# ==========================================
def find_symmetric_components(src_leaf, dst_leaf):
    # Dictionary 結構: 
    # Key: Signature (tuple) : the length of tuple is the link from src to dst
    # the element in the tuple is another tuple like : 'l*->l*_CF:(float)'. it mean all the path contain this link and its CF  
    # Value: {'paths': [list of path (a list)], 'weight': sum of the bandwidth of bandwidth of bottleneck link }
    components = {} 
    
    for path in nx.all_shortest_paths(G, source=src_leaf, target=dst_leaf):
        
        signature = []
        path_bottleneck = float('inf')
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            # sorted
            edge_labels = tuple(sorted(list(Q[u][v]['labels'])))
            signature.append(edge_labels)
            
            # find the bottleneck for weight
            path_bottleneck = min(path_bottleneck, G[u][v]['bw'])
            
        sig_key = tuple(signature)
        
        # grouping the path from sig_key and add bottleneck bandwidth to the weight
        if sig_key not in components:
            components[sig_key] = {'paths': [], 'weight': 0.0}
            
        components[sig_key]['paths'].append(path)
        components[sig_key]['weight'] += path_bottleneck
    print(components)
    return components


src, dst = 'l1', 'l2'
print(f"\n {src} to {dst} components")
components = find_symmetric_components(src, dst)

for idx, (sig_key, comp_info) in enumerate(components.items()):
    paths = comp_info['paths']
    weight = comp_info['weight']
    
    print(f"\n[ Component {idx+1} ]")
    print(f"  > path num : {len(paths)}")
    print(f"  > W-ECMP weight : {weight} ")
    for p in paths:
        print(f"  -> {' - '.join(p)}")