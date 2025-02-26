import csv
import numpy as np
#class for the graph the game is based of, loads graph from csv file

class Graph:
    def __init__(self, csv_file):
        # (x, y) position for each node save with node_id as key and (x, y) as tuple
        self.nodes = {}
        # dictionary for neighbors with node_id as key and neighbors as numpy array
        self.adjacency = {}
        # lists for start and end nodes for the sub
        self.start_nodes = {}
        self.end_nodes = {}
        # when initializing at end loads graph from csv file
        self.load_from_csv(csv_file)

    def load_from_csv(self, csv_file):
        # Expected columns: node_id,x,y,is_start,is_end,neighbors
        # Here we assume that after is_end, all subsequent fields are neighbors.
        with open(csv_file, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
            size = len(rows)
            for row in rows:
                # for each row saves the values, see structure of csv file above
                node_id = int(row[0])
                x = float(row[1])
                y = float(row[2])
                is_start = int(row[3])
                is_end = int(row[4])
                # remaining fields are neighbors
                neighbors = [int(n) for n in row[5:]]

                # new entry into dictionaries
                self.nodes[node_id] = (x, y)
                self.adjacency[node_id] = self.convert_to_mask(neighbors, size)

                self.start_nodes[node_id] = bool(is_start)
                self.end_nodes[node_id] = bool(is_end)

    def neighbors(self, node):
        return self.adjacency[node]
    
    def convert_to_mask(self, neighbors, size):
        # initialiserar en array med samma size som action space
        mask = np.zeros(size, dtype=bool)
        # sätter alla legala drag till True
        mask[neighbors] = True
        return mask
