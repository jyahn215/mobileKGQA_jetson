"""
following code is based on the below repository:
https://github.com/RManLuo/reasoning-on-graphs/blob/master/src/utils/graph_utils.py
"""

import networkx as nx
from collections import deque
import walker



def build_graph_as_nx_format(graph):
    G = nx.Graph()
    for head, relation, tail in graph:
        G.add_edge(head, tail, relation=relation)
    return G


def bfs_with_rule(graph, start_node, target_rule):
    result_paths = []
    queue = deque([(start_node, [])])
    if not (start_node in graph):
        raise ValueError("The start node does not exist in the graph.")
    while queue:
        current_node, current_path = queue.popleft()

        if len(current_path) == len(target_rule) and current_node != start_node:
            result_paths.append(current_path)

        if len(current_path) < len(target_rule):
            for neighbor in graph.neighbors(current_node):
                rel = graph[current_node][neighbor]['relation']
                if rel != target_rule[len(current_path)] or len(current_path) > len(target_rule):
                    continue
                queue.append((neighbor, current_path +
                             [(current_node, rel, neighbor)]))

    answer_list = []
    for path in result_paths:
        answer_list.append(path[-1][-1])

    return answer_list


def get_shortest_paths(q_entity: list, a_entity: list, graph: nx.Graph) -> list:
    '''
    Get shortest paths connecting question and answer entities.
    '''
    # Select paths
    paths = []
    for h in q_entity:
        if h not in graph:
            continue
        for t in a_entity:
            if t not in graph:
                continue
            try:
                for p in nx.all_shortest_paths(graph, h, t):
                    paths.append(p)
            except:
                pass
    # Add relation to paths
    result_paths = []
    for p in paths:
        tmp = []
        for i in range(len(p)-1):
            u = p[i]
            v = p[i+1]
            tmp.append((u, graph[u][v]['relation'], v))
        result_paths.append(tmp)
    return result_paths


def get_simple_paths(q_entity: list, a_entity: list, graph: nx.Graph, hop=2) -> list:
    '''
    Get all simple paths connecting question and answer entities within given hop
    '''
    # Select paths
    paths = []
    for h in q_entity:
        if h not in graph:
            continue
        for t in a_entity:
            if t not in graph:
                continue
            try:
                for p in nx.all_simple_edge_paths(graph, h, t, cutoff=hop):
                    paths.append(p)
            except:
                pass
    # Add relation to paths
    result_paths = []
    for p in paths:
        result_paths.append(
            [(e[0], graph[e[0]][e[1]]['relation'], e[1]) for e in p])
    return result_paths


def get_negative_paths(q_entity: list, a_entity: list, graph: nx.Graph, n_neg: int, hop=2) -> list:
    '''
    Get negative paths for question witin hop
    '''
    # sample paths
    start_nodes = []
    end_nodes = []
    node_idx = list(graph.nodes())
    for h in q_entity:
        if h in graph:
            start_nodes.append(node_idx.index(h))
    for t in a_entity:
        if t in graph:
            end_nodes.append(node_idx.index(t))
    paths = walker.random_walks(
        graph, n_walks=n_neg, walk_len=hop, start_nodes=start_nodes, verbose=False)
    # Add relation to paths
    result_paths = []
    for p in paths:
        tmp = []
        # remove paths that end with answer entity
        if p[-1] in end_nodes:
            continue
        for i in range(len(p)-1):
            u = node_idx[p[i]]
            v = node_idx[p[i+1]]
            tmp.append((u, graph[u][v]['relation'], v))
        result_paths.append(tmp)
    return result_paths


def get_random_paths(q_entity: list, graph: nx.Graph, n=3, hop=2) -> tuple[list, list]:
    '''
    Get negative paths for question witin hop
    '''
    # sample paths
    start_nodes = []
    node_idx = list(graph.nodes())
    for h in q_entity:
        if h in graph:
            start_nodes.append(node_idx.index(h))
    paths = walker.random_walks(
        graph, n_walks=n, walk_len=hop, start_nodes=start_nodes, verbose=False)
    # Add relation to paths
    result_paths = []
    rules = []
    for p in paths:
        tmp = []
        tmp_rule = []
        for i in range(len(p)-1):
            u = node_idx[p[i]]
            v = node_idx[p[i+1]]
            tmp.append((u, graph[u][v]['relation'], v))
            tmp_rule.append(graph[u][v]['relation'])
        result_paths.append(tmp)
        rules.append(tmp_rule)
    return result_paths, rules


def get_local_sequence(graph, start_node, max_hops):
    if start_node not in graph:
        raise ValueError("The start node does not exist in the graph.")

    visited = set()
    queue = deque([(start_node, 0)])
    sequence = [start_node]
    visited.add(start_node)

    while queue:
        current_node, current_hop = queue.popleft()

        if current_hop >= max_hops:
            continue

        for neighbor in graph.neighbors(current_node):
            if neighbor not in visited:
                relation = graph.get_edge_data(
                    current_node, neighbor)['relation']
                sequence.append((relation, neighbor))
                visited.add(neighbor)
                queue.append((neighbor, current_hop + 1))

    return sequence


def check_validity(tokenizer, word, thr=0.5):
    return len(tokenizer.tokenize(word)) / len(word) < thr


def sample_valid_path(graph, length, tokenizer, args):

    iter_cnt = 0
    while True:
        if iter_cnt > 20:
            return None
        else:
            iter_cnt += 1
        # path sampling
        ent_in_path = nx.generate_random_paths(
            graph, sample_size=1, path_length=length)

        # check path validity
        try:
            ent_in_path = next(ent_in_path)
        except:
            return None  # empty graph

        rel_in_path = []
        for idx in range(len(ent_in_path)-1):
            h = ent_in_path[idx]
            t = ent_in_path[idx+1]
            rel_in_path.append(graph[h][t]['relation'])

        if len(set(rel_in_path)) != len(rel_in_path):  # duplicate in path
            continue
        if len(ent_in_path) != length + 1:  # wrong graph (ex. no edges)
            return None
        if len(set(ent_in_path)) != len(ent_in_path):  # duplicate in path
            continue

        # filter out invalid entities
        if args.method == "mobileKGQA":
            valid = True
            for ent in ent_in_path:
                if check_validity(tokenizer, ent) is False:
                    valid = False
                    break
                if len(ent) > 100:
                    valid = False
                    break
            if valid is False:
                continue
        ent_in_path = [str(ent)
                       for ent in ent_in_path]  # avoid type conflict error

        # generate path
        reasoning_path = [ent_in_path[0]]
        for i in range(len(ent_in_path)-1):
            h = ent_in_path[i]
            t = ent_in_path[i+1]
            r = graph[h][t]['relation']
            reasoning_path.append(r)
            reasoning_path.append(t)

        return reasoning_path
