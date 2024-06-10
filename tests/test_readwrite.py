"""
Test reading and writing of graphs.
"""
import os
from graph_states import Graph, GraphFactory

def test_graph1():
    """
    Test writing graph as TGF
    """
    filepath = 'tmp.tgf'
    graph1 = GraphFactory.get_star_graph(6, 4, 1)
    graph2 = GraphFactory.get_star_graph(6, 4, 0)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(graph1.to_tgf())
    with open(filepath, 'r', encoding='utf-8') as f:
        graph3 = Graph.from_tgf(filepath)
        assert graph3 == graph1
        assert graph3 != graph2
    if os.path.exists(filepath):
        os.remove(filepath)
