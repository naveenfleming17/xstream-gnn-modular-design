import dgl
import torch
import ast

def build_graph_from_csv(nodes_df, site_row):
    src_nodes, dst_nodes = [], []
    node_features, edge_features, edge_weights = [], [], []
    edge_set = set()

    for _, node_row in nodes_df.iterrows():
        node_features.append([
            node_row['num_children'], node_row['x_coordinate'], node_row['y_coordinate'],
            node_row['z_coordinate'], node_row['area'], node_row['volume'],
            node_row['avg_normals_x'], node_row['avg_normals_y'], node_row['avg_normals_z'],
            node_row['bb_depth'], node_row['bb_width'], node_row['bb_height'],
            node_row['surface_area'], node_row['density'],
            node_row['vertices_faces_ratio'], node_row['vertices_edges_ratio']
        ])

        src = node_row['node_id']
        children = ast.literal_eval(str(node_row['children_id']))
        angles = ast.literal_eval(str(node_row['angle_node_children']))
        lengths = ast.literal_eval(str(node_row['edge_length']))

        for child, angle, length in zip(children, angles, lengths):
            edge = (src, child)
            reverse = (child, src)
            if edge not in edge_set and reverse not in edge_set:
                src_nodes.append(src)
                dst_nodes.append(child)
                edge_features.append([angle, length])
                edge_weights.append(length)
                edge_set.add(edge)
                edge_set.add(reverse)

    g = dgl.graph((src_nodes, dst_nodes))

    g.ndata['feature'] = torch.tensor(node_features, dtype=torch.float32)
    g.edata['feature'] = torch.tensor(edge_features, dtype=torch.float32)
    g.edata['weight'] = torch.tensor(edge_weights, dtype=torch.float32)

    return g
