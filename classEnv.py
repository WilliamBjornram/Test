import gymnasium
from gymnasium.spaces import Discrete, MultiBinary
import random
import numpy as np
import tkinter as tk
import time

#Testtesttest

class GameEnv(gymnasium.Env):
    def __init__(self, graph, mode='None'):
        super(GameEnv, self).__init__()
        self.graph = graph

        # action space är hela grafen
        max = len(self.graph.nodes)
        self.action_space = Discrete(max)
        # vi lägger till två element på slutet för is_start och is_end
        self.observation_space = MultiBinary(max+2)

        # ett spel får inte vara längre än 100 drag
        self.game_length = 100

        # randomiserar vart ubåten startar
        self.state = random.choice([key for key, value in self.graph.start_nodes.items() if value == 1])

        # vilka noder att patrullera för båten
        self.patrol_nodes = [key for key, value in self.graph.start_nodes.items() if value == 0]
        # randomiserar vart båten startar
        self.boat_pos = random.choice(self.patrol_nodes)
        # initialiserar en dict för att se hur många gånger som besökt respektive nod
        self.visited_count = {node: 0 for node in self.patrol_nodes}

        # init GUI if render mode = human
        self.mode = mode
        if self.mode == 'human':
            self.GUI_init()
            self.gui_initialized = True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # startar om spelet, reset på timer
        self.game_length = 100

        # bestämmer vart ubåten startar
        self.state = random.choice([key for key, value in self.graph.start_nodes.items() if value == 1])

        # bestämmer vart båten startar
        self.boat_pos = random.choice(self.patrol_nodes)

        # initialiserar om besökt dict för båt
        self.visited_count = {node: 0 for node in self.patrol_nodes}

        if self.mode == 'human':
            self.render()

        # fetch observationer, info tom
        obs = self._get_obs()
        info = {}

        #förvantat av gymnasium att vi ska returnera
        return obs, info

    def step(self, action):
        # ett drag gjort
        self.game_length -= 1

        # innan ändrar state, kollar så att vald action returnerar True och alltså är giltig granne
        if self.graph.adjacency[self.state][action]:
            self.state = int(action)
        else:
            raise Exception("Illegal move attempted")
        
        if self.mode == 'human':
            self.render()
        
        # rör båten
        self.boat_pos = self.calc_move(self.boat_pos)

        if self.mode == 'human':
            self.render()

        # om ubåten står på en slut nod -> reward = 1
        # annars om ubåt.pos == båt.pos -> reward = -1
        if self.graph.end_nodes[self.state]:
            reward = +1
        elif self.state == self.boat_pos:
            reward = -1
        else:
            reward = +0

        # kollar om klar med spelet
        if self.game_length == 0 or reward != 0:
            done = True
        else:
            done = False
        
        # fetch observationer, info tom
        obs = self._get_obs()
        info = {}

        #förvantat av gymnasium att vi ska returnera
        return obs, reward, done, False, info
    
    # observation ser ut som np.array[neighbors, start_node, end_node]
    def _get_obs(self):
        start = self.graph.start_nodes[self.state]
        end = self.graph.end_nodes[self.state]
        obs = np.append(self.graph.adjacency[self.state], [start, end])
        return obs
    
    # returnerar en action mask för att algoritm ska veta vilka drag som giltiga och ej
    def action_mask(self):
        return self.graph.adjacency[self.state]

    # beräkna nästa drag för båten
    def calc_move(self, current_position):

        # markerar den nuvarande positionen som besökt
        self.visited_count[current_position] += 1

        # hitta den till antalet minst antal besök till en eller flera noder
        min_visits = min(self.visited_count.values())

        # hitta alla noder som blivit besökta minst antal gånger
        least_visited_nodes = [node for node, count in self.visited_count.items() if count == min_visits]

        # randomisera nästa drag bland de noder som blivit besökta minst antal gånger
        next_position = random.choice(least_visited_nodes)

        return next_position
    
    def GUI_init(self):
        self.master = tk.Tk()
        self.master.title("The Sub and The Boat Game")

        # dimensioner för canvas
        self.canvas_width = 1000
        self.canvas_height = 600
        self.canvas = tk.Canvas(self.master, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack(side=tk.TOP, padx=10, pady=10)

        self.status_label = tk.Label(self.master, text="")
        self.status_label.pack(side=tk.TOP, pady=5)

        self.node_tags = {}

        # skala grafen till canvas
        self.scale_graph_to_canvas()

        # rita up grafen
        self.draw_graph()


    # för GUI
    def render(self, mode='human'):

        if mode == 'human':
            if not hasattr(self, 'gui_initialized') or not self.gui_initialized:
                self.mode = mode
                self.GUI_init()
                self.gui_initialized = True

        for node, oval in self.node_tags.items():
            color = "lightblue"
            width = 2

            if node == self.state and node == self.boat_pos:
                color = "purple"
                width = 3
            elif node == self.state:
                color = "darkblue"
                width = 3
            elif node == self.boat_pos:
                color = "orange"
                width = 3

            self.canvas.itemconfig(oval, fill=color, outline="blue", width=width)

        self.status_label.config(text=f"The Sub at {self.state}, The Boat at {self.boat_pos}")

        # uppdatera och sov sen i 2 sek, så hinner uppfatta vad som händer
        self.master.update()
        time.sleep(2.0)

    def close(self):
        if hasattr(self, 'master') and self.master is not None:
            self.master.destroy()
            self.master = None

    def scale_graph_to_canvas(self):
        # få min och max för x och y från originella grafen
        xs = [coord[0] for coord in self.graph.nodes.values()]
        ys = [coord[1] for coord in self.graph.nodes.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # har en marginal så att det inte ritas direkt på gränsen av canvas
        margin = 50

        # beräknar höjd och bredd för originella grafen
        width = max_x - min_x
        height = max_y - min_y

        # beräknar hur mycket utrymme vi har att rita på i canvas
        available_width = self.canvas_width - 2 * margin
        available_height = self.canvas_height - 2 * margin

        # beräknar faktor att skala graferna med för att passa in snyggt i canvas
        scale_factor = min(available_width / width if width > 0 else float('inf'),
                           available_height / height if height > 0 else float('inf'))

        # beräknar center av originella grafen
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        # beräknar center för canvas
        canvas_center_x = self.canvas_width / 2
        canvas_center_y = self.canvas_height / 2

        # beräknar hur mycket att skifta för att centrera grafen i canvas
        shift_x = canvas_center_x - scale_factor * center_x
        shift_y = canvas_center_y - scale_factor * center_y

        # applicera ovan beräkningar på nodernas x,y värden
        for node_id, (x, y) in self.graph.nodes.items():
            new_x = x * scale_factor + shift_x
            new_y = y * scale_factor + shift_y
            self.graph.nodes[node_id] = (new_x, new_y)

    def draw_graph(self):
        # rita linjer
        for node, coords in self.graph.nodes.items():
            x, y = coords
            adj_l = np.where(self.graph.adjacency[node])[0].tolist()
            for nbr in adj_l:
                nx, ny = self.graph.nodes[nbr]
                if node < nbr:
                    self.canvas.create_line(x, y, nx, ny, fill="black")
        
        # riter alla noder som cirklar
        for node, (x, y) in self.graph.nodes.items():
            r = 15
            oval = self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="lightblue", outline="blue", width=2)
            self.canvas.create_text(x, y, text=str(node), fill="black")
            self.node_tags[node] = oval