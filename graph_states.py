"""
Some code for generating graph states.
"""
import math
import random
import itertools
import networkx as nx

class Graph:
    """
    Graphs are represented by a map : (v, w) --> val.
    """

    @staticmethod
    def all_possible_edges(n: int):
        """
        Get all combinations of nodes w/o duplicates, e.g. (0,1) but not (1,0).
        """
        return itertools.combinations(range(n), 2)

    @staticmethod
    def random_edges(n: int, k: int):
        """
        Randomly get k edges from all possible edges from an `n` node graph.
        """
        return random.sample(list(Graph.all_possible_edges(n)), k)

    @staticmethod
    def edge_equal(e1, e2):
        """
        Check if two (undirected) edges are equal. (a,b) and (b,c) are 
        considered equal.
        """
        return min(e1) == min(e2) and max(e1) == max(e2)
    
    @staticmethod
    def from_tgf(filepath: str):
        """
        Load a graph from a .tgf file.
        """
        max_node = 0
        edges = set()
        with open(filepath, 'r', encoding='utf-8') as f:
            reading_edges = False
            lines = f.readlines()
            for line in lines:
                tokens = line.split()
                if (reading_edges):
                    # TGF starts numbering nodes at 1, we start at 0
                    v = int(tokens[0])-1
                    w = int(tokens[1])-1
                    edges.add((v, w))
                    max_node = max(max_node, v, w)
                elif tokens[0] == "#":
                    # read until '#' is a starting character (skip all node names)
                    reading_edges = True
        graph = Graph(max_node+1, name=filepath)
        for v, w in edges:
            graph.set_edge(v, w, True)
        return graph


    def __init__(self, num_nodes: int, name: str = '', from_graph = None):
        """
        If from_graph is not set, inits an empty graph, i.e. a graph where every
        edge is set to False. If from_graph is set, creates a new graph which
        is a (deep) copy of from_graph.
        """
        self._num_nodes = num_nodes
        self.name = name
        self.edges = {}

        if from_graph is None:
            for v, w in Graph.all_possible_edges(num_nodes):
                self.edges[(v, w)] = False
        else:
            for v, w in Graph.all_possible_edges(num_nodes):
                self.edges[(v, w)] = from_graph.get_edge(v, w)


    @property
    def num_nodes(self):
        """
        Number of nodes of the graph.
        """
        return self._num_nodes


    def get_non_isolated_nodes(self):
        """
        Get all nodes which have at least connected edge.
        """
        nodes = set()
        for (u, v) in Graph.all_possible_edges(self.num_nodes):
            if self.get_edge(u, v) is True:
                nodes.add(u)
                nodes.add(v)
        return nodes


    def get_isolated_nodes(self):
        """
        Get all nodes which are not connected to anything.
        """
        nodes = set(range(self._num_nodes))
        return nodes - self.get_non_isolated_nodes()


    def set_edge(self, v: int, w: int, val=True):
        """
        Set edge (v, w) to val.
        """
        if v == w:
            raise ValueError("Self loops are not allowed.")
        self.edges[(min(v, w), max(v, w))] = val

    def get_edge(self, v: int, w: int):
        """
        Returns True iff the graph self has an edge between given nodes.
        """
        if v == w:
            raise ValueError("Self loops are not allowed.")
        return self.edges[(min(v, w), max(v, w))]

    def get_neighborhood(self, u: int):
        """
        Get the neighborhood of node v.
        """
        neigh_u = []
        for (v, w) in Graph.all_possible_edges(self.num_nodes):
            if self.edges[(v, w)]:
                if v == u:
                    neigh_u.append(w)
                if w == u:
                    neigh_u.append(v)
        return neigh_u

    def get_adjacency_lists_flattened(self):
        """
        Returns the adjacency lists for each node as a single flattened list.
        """
        ret = []
        for v, w in Graph.all_possible_edges(self.num_nodes):
            ret.append(self.edges[(v, w)])
        return ret

    def to_tgf(self):
        """
        Return a string representation of the graph in TGF format.
        """
        tgf = "#\n"
        for v, w in Graph.all_possible_edges(self.num_nodes):
            if self.edges[(v, w)]:
                # TGF starts numbering nodes at 1, we start at 0
                tgf += f"{v+1} {w+1}\n"
        return tgf

    def __str__(self):
        return str(self.edges)
    
    def __eq__(self, other):
        for v, w in Graph.all_possible_edges(self.num_nodes):
            if self.get_edge(v, w) != other.get_edge(v, w):
                return False
        return True



class GraphFactory:
    """
    Functions to generate different graphs.
    """

    @staticmethod
    def get_complete_graph(n: int):
        """
        Generate a complete graph.
        """
        graph = Graph(n)
        for (v, w) in Graph.all_possible_edges(n):
            graph.set_edge(v, w, True)
        return graph

    @staticmethod
    def get_empty_graph(n: int):
        """
        Generate an empty graph.
        """
        graph = Graph(n)
        return graph

    @staticmethod
    def get_star_graph(n: int, star_vertices: int | str = 'all', center: int = 0):
        """
        Generate a star graph with the given number of vertices.
        """
        if star_vertices == 'all':
            star_vertices = n
        if star_vertices > n:
            raise ValueError("star_vertices should be <= n")
        graph = Graph(n, name=f'GHZ({star_vertices})')
        for v in range(star_vertices):
            if v != center:
                graph.set_edge(center, v, True)
        return graph

    @staticmethod
    def get_erdos_renyi_graph(n: int, p: float, seed = None):
        """
        Generate an Erdos-Renyi random graph.
        """
        if not seed is None:
            random.seed(seed)
        graph = Graph(n, name=f'ER({n};{p})')
        for (v, w) in Graph.all_possible_edges(n):
            if random.random() < p:
                graph.set_edge(v, w, True)
        return graph

    @staticmethod
    def get_dist_hereditary_graph(n: int, p: float = 1/3, seed = None):
        """
        Generate a random distance-hereditary graph. (Dahlberg '20. sec. 2.5.1)
        p is the probability of adding a leaf, and 1-p the probability of doing
        a twin split.
        """
        if not seed is None:
            random.seed(seed)

        graph = Graph(n, name=f'DH({n};{p})')
        nodes = [0] # start with a graph with only 1 node (0)
        for v in range(1, n):
            u = random.choice(nodes) # pick a random node u from existing nodes
            nodes.append(v) # add new node v

            # Randomly choose add leaf or twin split
            if random.random() < p:
                # Add a leaf
                graph.set_edge(u, v, True)
            else:
                # Twin-split
                neigh_u = graph.get_neighborhood(u)
                if random.random() < 0.5:
                    # False twin-split
                    for x in neigh_u:
                        graph.set_edge(v, x, True)
                else:
                    # True twin-split
                    for x in neigh_u:
                        graph.set_edge(v, x, True)
                    graph.set_edge(v, u, True)

        return graph

    @staticmethod
    def get_rabbie2022_network(which: str, p: float = 0):
        """
        Return the network from "Designing quantum networks using preexisting
        infrastructure" Rabbie et al. (2022) Figure 3.

        Args:
            which in ['random', 'gzh']
        """
        num_nodes = 14
        names = {'Almere' : 0, 'Amsterdam_2' : 1, 'Arnhem' : 2, 'Delft_1' : 3,
                 'Dwingeloo' : 4, 'Eindhoven_1' : 5, 'Enschede_2' : 6 , 
                 'Groningen_1' : 7, 'Maastricht' : 8, 'Meppel' : 9, 
                 'Nieuwegein' : 10, 'Venlo' : 11, 'Zwolle_1' : 12,
                 'Zwolle_2' : 13}
        names_inv = {val : key for key, val in names.items()}

        if which.lower() == 'ghz':
            # 4-qubit GHZ state between main nodes
            graph = Graph(num_nodes, name='RABBIE_GHZ')
            graph.set_edge(names['Delft_1'], names['Groningen_1'], True)
            graph.set_edge(names['Delft_1'], names['Enschede_2'], True)
            graph.set_edge(names['Delft_1'], names['Maastricht'], True)
            return graph
        elif which.lower() == 'random':
            # random entanglement with prob p^dist between nodes
            network = nx.Graph()
            network.add_edge('Groningen_1', 'Dwingeloo')
            network.add_edge('Dwingeloo', 'Meppel')
            network.add_edge('Meppel', 'Zwolle_1')
            network.add_edge('Zwolle_1', 'Zwolle_2')
            network.add_edge('Zwolle_1', 'Enschede_2')
            network.add_edge('Zwolle_1', 'Arnhem')
            network.add_edge('Arnhem', 'Venlo')
            network.add_edge('Venlo', 'Maastricht')
            network.add_edge('Maastricht', 'Eindhoven_1')
            network.add_edge('Eindhoven_1', 'Nieuwegein')
            network.add_edge('Nieuwegein', 'Delft_1')
            network.add_edge('Nieuwegein', 'Almere')
            network.add_edge('Delft_1', 'Amsterdam_2')
            network.add_edge('Delft_1', 'Almere')
            network.add_edge('Amsterdam_2', 'Almere')
            network.add_edge('Almere', 'Zwolle_2')
            dists = dict(nx.all_pairs_shortest_path_length(network))

            graph = Graph(num_nodes, name=f'RABBIE_RAND({p})')
            for u, v in Graph.all_possible_edges(num_nodes):
                dist = dists[names_inv[u]][names_inv[v]]
                if random.random() < p**dist:
                    graph.set_edge(u, v, True)
            return graph
        else:
            raise ValueError("Argument 'which' should be in ['random', 'ghz']")
