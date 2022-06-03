from nn import Gene, Network
import pygame
import nn
import random
import numpy as np
import time
import joblib
import math

import networkx
import matplotlib.pyplot as plt

# inputs: {last move X, last move Y, north/south border dist, east/west border dist, location X, location Y, nearest border distance}
NUM_SENSORY_INPUTS = 15
nn.set_num_input(NUM_SENSORY_INPUTS)
nn.set_num_hidden(8)


GENOME_SIZE = 32
nn.set_genome_size(GENOME_SIZE)


class Actions:
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_RIGHT = 2
    MOVE_LEFT = 3
    MOVE_RAND = 4
    MOVE_FORWARD = 5
    MOVE_BACKWARD = 6
    STAND_STILL = 7
    RELEASE_PHEROMONE = 8
    END = 9
    
class Directions:
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3 
    
nn.set_num_output(Actions.END)

# TODO move elsewhere
class Agent:
    def __init__(self, genome: list[int], loc: tuple[int]):
        self.brain = Network.from_genome(genome)
        self.location = loc
        self.input_data = None
        self.last_move = (0, 0)
        self.forward_direction = random.choice([Directions.UP, Directions.DOWN, Directions.LEFT, Directions.RIGHT])
    
    def get_genome(self):
        return self.brain.genome
    
    def get_forward(self):
        return self.forward_direction
    
    def get_color(self):
        color_sum = sum(self.brain.genome)
        color = color_sum % 16777215
        
        r = color & 0xFF
        g = (color >> 8) & 0xFF
        b = (color >> 16) & 0xFF
        return (r, g, b)
    
    def draw_brain(self):
        G = networkx.DiGraph()
        
        edge_list = []
        
        input_names = ["LstMvX", "LstMvY", "NSDist", "EWDist", "EWPos", 
                   "NSPos", "NearBord", "NOcc", "SOcc", "EOcc", 
                   "WOcc", "FwdLook", "Pher", "GenSim", "Osc", "Bias"]
        
        output_names = ["MvUp", "MvDown", "MvRight", "MvLeft", "MvRand", "MvFwd", "MvBck", "Still",  "RelPher"]
        
        # input --> hidden
        hidden_index: int = 0
        for hidden_row in self.brain.weight_layers[0]:
            edge_list += [(input_names[i], f"H{hidden_index}") for i in range(len(hidden_row)) if hidden_row[i] != 0]
            hidden_index += 1
        
        # hidden --> output
        output_index: int = 0
        for output_row in self.brain.weight_layers[1]:
            edge_list += [(f"H{i}", output_names[output_index]) for i in range(len(output_row)) if output_row[i] != 0]
            output_index += 1
            
        # hidden --> hidden
        hidden_index: int = 0
        for hidden_row in self.brain.weight_layers[2]:
            edge_list += [(f"H{i}", f"H{hidden_index}") for i in range(len(hidden_row)) if hidden_row[i] != 0]
            hidden_index += 1
            
        # input --> output
        output_index: int = 0
        for input_row in self.brain.weight_layers[3]:
            edge_list += [(input_names[i], output_names[output_index]) for i in range(len(input_row)) if input_row[i] != 0]
            hidden_index += 1
        
        G.add_edges_from(edge_list)
        
        
        
        val_map = { "LstMvX" : .25, "LstMvY" : .25, "NSDist" : .25, "EWDist" : .25, "EWPos" : .25, 
                   "NSPos" : .25, "NearBord" : .25, "NOcc" : .25, "SOcc" : .25, "EOcc" : .25, 
                   "WOcc" : .25, "FwdLook" : .25, "Pher" : .25, "GenSim" : .25, "Osc" : .25, "Bias" : .25,
                   "MvUp" : 0.75, "MvDown" : 0.75, "MvRight" : 0.75, "MvLeft" : 0.75, "MvRand" : 0.75, "MvFwd" : 0.75, "MvBck" : 0.75, "Still" : 0.75,  "RelPher" : 0.75
                   }
        
        node_color_values = [val_map.get(node, 0.5) for node in G.nodes()]
        
        pos = networkx.multipartite_layout(G)
        networkx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_color=node_color_values, node_size=250)
        networkx.draw_networkx_labels(G, pos)
        networkx.draw_networkx_edges(G, pos, edgelist=edge_list, arrows=False)
        
        plt.show()
    
    def get_x(self):
        return self.location[0]

    def set_x(self, new_x):
        if self.get_loc() is not None:
            self.last_move = (new_x - self.get_x(), self.get_y())
            if self.last_move[0] == 1:
                self.forward_direction = Directions.RIGHT
            elif self.last_move[0] == -1:
                self.forward_direction = Directions.LEFT
        self.location = (new_x, self.get_y())
        
    def get_y(self):
        return self.location[1]
    
    def set_y(self, new_y):
        if self.get_loc() is not None:
            self.last_move = (self.get_x(), new_y - self.get_y())
            if self.last_move[1] == -1:
                self.forward_direction = Directions.DOWN
            elif self.last_move[1] == 1:
                self.forward_direction = Directions.UP
        self.location = (self.get_x(), new_y)
        
    def set_loc(self, new_loc: tuple[int]):
        self.set_x(new_loc[0])
        self.set_y(new_loc[1])
        
    def get_loc(self):
        return self.location
    
    def get_last_move(self):
        return self.last_move
    
    def set_input_data(self, inputs):
        self.input_data = inputs
    
    def think(self):
        assert self.input_data is not None, "did you forget to set input data?"
        
        outputs = self.brain.feed_forward(self.input_data)
        
        self.input_data = None
        
        return outputs

SCREEN_WIDTH = 768
SCREEN_HEIGHT = 768
        
class World:
    PHEROMONE_STEP_DURATION = 10
    PHEROMONES_ENABLED = False
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.TILE_WIDTH = SCREEN_WIDTH / self.width
        self.TILE_HEIGHT = SCREEN_HEIGHT / self.height 
        
        self.cur_step: int = 0
        self.cur_gen: int = 0
        
        if World.PHEROMONES_ENABLED:
            self.pheromones: list[tuple[tuple[int,int], int, int]] = []
        self.agents = []
        self.map = []
        for y in range(self.height):
            self.map.append([])
            for _ in range(self.width):
                self.map[y].append(None)
    
    def get_random_open_tile(self):
        iters: int = 0
        while True:
            loc = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            if self.map[loc[1]][loc[0]] is None:
                return loc
            iters += 1
            if iters > 1000:
                assert False, "something went wrong!"
    
    def create_wall(self, x: int, y: int, width: int, height: int):
        for yi in range(y, y + height):
            for xi in range(x, x + width):
                self.map[yi][xi] = False
    
    def reset(self):
        self.clear_map()
        self.cur_step: int = 0
        self.cur_gen += 1
        if World.PHEROMONES_ENABLED:
            self.pheromones = []
    
    def clear_map(self, clear_agents: bool = True):
        if clear_agents:
            self.agents = []
            
        for y in range(self.height):
            for x in range(self.width):
                if self.map[y][x] is not False:
                    self.map[y][x] = None
        
    def set_agent_at_tile(self, agent: Agent, loc: tuple[int]):
        assert self.is_in_map(loc), "location is out of map bounds!"
        assert self.is_open_tile(loc), "tile='{loc[0]}, {loc[1]}' is not empty!"
             
        self.map[loc[1]][loc[0]] = agent
    
    def clear_tile(self, loc: tuple[int]):
        assert self.is_in_map(loc), "tile is out of map bounds!"
        self.map[loc[1]][loc[0]] = None

    def is_in_map(self, loc: tuple[int]):
        return loc[0] >= 0 and loc[0] < self.width and loc[1] >= 0 and loc[1] < self.height
    
    def is_open_tile(self, loc: tuple[int]) -> bool:
        return self.is_in_map(loc) and self.map[loc[1]][loc[0]] is None
    
    def move_agent_to(self, agent, new_loc: tuple[int]):
        assert self.is_in_map(new_loc), "location is out of map bounds!"
        assert self.is_open_tile(new_loc), "tile is not free!"
        self.set_agent_at_tile(agent, new_loc)
        self.clear_tile(agent.get_loc())
        agent.set_loc(new_loc)

    def create_random_agents(self, num_agents):
        self.agents = [None]*num_agents
        
        for i in range(num_agents):
            agent = Agent(Gene.create_random_genome(GENOME_SIZE), self.get_random_open_tile())
            self.agents[i] = agent
            self.set_agent_at_tile(agent, agent.get_loc())
        
            
    def set_agents(self, agents, random_location: bool = True):
        self.agents = agents
        
        if random_location:
            for agent in self.agents:
                # don't use agent.set_loc() because agent.loc = None
                agent.location = self.get_random_open_tile()
                self.set_agent_at_tile(agent, agent.get_loc())

    @staticmethod
    def create_new_world(width: int, height: int, num_agents: int = None):
        world = World(width, height)
        
        #world.create_wall(75, 33, 10, 66)
        
        if num_agents is not None:
            world.create_random_agents(num_agents)
            
        
        return world
    
    @staticmethod
    def create_new_generation(agents, did_survive_func, mutation_chance: float = 0.001, max_pop_size: int = None):
        next_generation: list[Agent] = []
        
        survived: int = 0
        for agent in agents:
            if did_survive_func(agent):
                for _ in range(random.randint(2, 5)):
                    new_genome = [gene if random.randint(0, 1000) < 1000 - (1000 * mutation_chance) else Gene.mutate(gene) for gene in agent.get_genome()]
                    next_generation.append(Agent(new_genome, loc=None))
                survived += 1
        
        print(f"{survived} agents survived ({(survived / max_pop_size * 100):.2f}%)")
                
        if len(next_generation) < max_pop_size:
            for _ in range(max_pop_size - len(next_generation)):
                next_generation.append(Agent(Gene.create_random_genome(GENOME_SIZE), loc=None))
        
        if len(next_generation) > max_pop_size:
            random.shuffle(next_generation)
            next_generation = next_generation[:max_pop_size]
            
        diversity_map = {}
        max_agent = None
        max_val: int = 0
        for agent in next_generation:
            diversity_map[sum(agent.brain.genome)] = diversity_map.get(sum(agent.brain.genome), 0) + 1
            if diversity_map[sum(agent.brain.genome)] > max_val:
                max_agent = agent
                max_val = diversity_map[sum(agent.brain.genome)]
        
        print(f"There is {(len(diversity_map.keys()) / len(next_generation)) * 100:0.2f}% genetic diversity")
        
        max_agent.draw_brain()
        
        return next_generation
    
    def draw(self, screen):
        
        for agent in self.agents:
            x = agent.get_x() * self.TILE_WIDTH
            y = agent.get_y() * self.TILE_HEIGHT
            pygame.draw.rect(screen, agent.get_color(), pygame.Rect(x, y, self.TILE_WIDTH, self.TILE_HEIGHT))
            
        for y in range(self.height):
            for x in range(self.width):
                if self.map[y][x] is False:
                    pygame.draw.rect(screen, (220,220,220), pygame.Rect(x * self.TILE_WIDTH, y * self.TILE_HEIGHT, self.TILE_WIDTH, self.TILE_HEIGHT))
    
    def get_north_south_border_dist(self, agent):
        return agent.get_y() - (self.height / 2)
    
    def get_east_west_border_dist(self, agent):
        return agent.get_x() - (self.width / 2)
    
    def get_agent_blocked_in_direction(self, agent, direction, look_ahead: int = 10):
        x = agent.get_x()
        y = agent.get_y()
        for i in range(look_ahead):
            # TODO could be written better
            if direction == Directions.RIGHT:
                if not self.is_open_tile((x + i, y)):
                    return i / look_ahead
            elif direction == Directions.LEFT:
                if not self.is_open_tile((x - i, y)):
                    return i / look_ahead
            elif direction == Directions.UP:
                if not self.is_open_tile((x, y + i)):
                    return i / look_ahead
            elif direction == Directions.DOWN:
                if not self.is_open_tile((x, y - i)):
                    return i / look_ahead
                
        return 1.0
    
    def get_nearest_border_dist_to_agent(self, agent):
        dist_right = self.width - agent.get_x()
        dist_left = agent.get_x()
        dist_up = agent.get_y()
        dist_down = self.height - agent.get_y()
        return min([dist_right, dist_left, dist_up, dist_down])
    
    def get_neareast_pheromone_to_agent(self, agent):
        # TODO assuming that width == height
        assert self.width == self.height, "OOPS NEED TO FIX"
        min_dist = self.width
        
        x1, y1 = agent.get_loc()
        angle = None
        for loc,_,_ in self.pheromones:
            x2, y2 = loc
            dist = abs(x1 - x2)**2 + abs(y1 - y2)**2
            min_dist = min(dist, min_dist)
            
        return min_dist

    def get_similarity_of_forward_neighbor(self, agent):
        x, y = agent.get_loc()
        mx, my = 0, 0
        dir = agent.get_forward()
        if dir == Directions.DOWN:
            my = -1
        elif dir == Directions.UP:
            my = 1
        elif dir == Directions.RIGHT:
            mx = 1
        elif dir == Directions.LEFT:
            mx = -1
        
        if not self.is_in_map((x + mx, y + my)) or self.map[y + my][x + mx] == False or self.map[y + my][x + mx] == None:
            return 0.0
        
        other_agent = self.map[y + my][x + mx]
        assert type(other_agent) is Agent
        
        dif = abs(sum(agent.brain.genome) - sum(other_agent.brain.genome))
        return dif / (0xFFFFFFFF * GENOME_SIZE)
        
        
        
        
    
    def step(self):
        
        cur = time.time()
        # inputs: {last move X, last move Y, north/south border dist, east/west border dist, location X, location Y, nearest border distance, blockage up, blockage down, blockage right, blockage left}
        input_data = np.zeros(NUM_SENSORY_INPUTS + 1, dtype=np.float32)
        
        for i, agent in enumerate(self.agents):
            # Gather input data
            
            
            input_data[0] = agent.get_last_move()[0]
            input_data[1] = agent.get_last_move()[1]
            input_data[2] = self.get_north_south_border_dist(agent) / self.height
            input_data[3] = self.get_east_west_border_dist(agent) / self.width
            input_data[4] = agent.get_loc()[0] / self.width
            input_data[5] = agent.get_loc()[1] / self.height
            
            # TODO assuming that width == height
            assert self.width == self.height, "OOPS NEED TO FIX"
            input_data[6] = self.get_nearest_border_dist_to_agent(agent) / self.width
            
            input_data[7] = 0.0 if self.is_open_tile((agent.get_x(), agent.get_y() + 1)) else 1.0
            input_data[8] = 0.0 if self.is_open_tile((agent.get_x(), agent.get_y() - 1)) else 1.0
            input_data[9] = 0.0 if self.is_open_tile((agent.get_x() + 1, agent.get_y())) else 1.0
            input_data[10] = 0.0 if self.is_open_tile((agent.get_x() - 1, agent.get_y())) else 1.0
            input_data[11] = self.get_agent_blocked_in_direction(agent, agent.get_forward(), 20) 
            
            # TODO assuming that width == height
            assert self.width == self.height, "OOPS NEED TO FIX"
            input_data[12] = self.get_neareast_pheromone_to_agent(agent) / self.width if World.PHEROMONES_ENABLED else 0.0
            
            input_data[13] = self.get_similarity_of_forward_neighbor(agent)
            
            input_data[14] = math.sin(self.cur_step / 3)
            
            input_data[-1] = 1.0
            
            #agent.set_input_data(input_data)
            
            # ===========
            # AGENT THINK
            # ===========
            
            outputs = agent.brain.feed_forward(input_data)
            
            # ===========
            
            thought = np.argmax(outputs)
            
            if thought == Actions.MOVE_UP:
                new_loc = (agent.get_x(), agent.get_y() + 1)
                if self.is_open_tile(new_loc):
                    self.move_agent_to(agent, new_loc)
            elif thought == Actions.MOVE_DOWN:
                new_loc = (agent.get_x(), agent.get_y() - 1)
                if self.is_open_tile(new_loc):
                    self.move_agent_to(agent, new_loc)
            elif thought == Actions.MOVE_RIGHT:
                new_loc = (agent.get_x() + 1, agent.get_y())
                if self.is_open_tile(new_loc):
                    self.move_agent_to(agent, new_loc)
            elif thought == Actions.MOVE_LEFT:
                new_loc = (agent.get_x() - 1, agent.get_y())
                if self.is_open_tile(new_loc):
                    self.move_agent_to(agent, new_loc)
            elif thought == Actions.MOVE_RAND:
                dir = random.choice([Directions.DOWN, Directions.UP, Directions.LEFT, Directions.RIGHT])
                if dir == Directions.UP:
                    new_loc = (agent.get_x(), agent.get_y() - 1)
                    if self.is_open_tile(new_loc):
                        self.move_agent_to(agent, new_loc)
                elif dir == Directions.DOWN:
                    new_loc = (agent.get_x(), agent.get_y() + 1)
                    if self.is_open_tile(new_loc):
                        self.move_agent_to(agent, new_loc)
                elif dir == Directions.RIGHT:
                    new_loc = (agent.get_x() + 1, agent.get_y())
                    if self.is_open_tile(new_loc):
                        self.move_agent_to(agent, new_loc)
                elif dir == Directions.LEFT:
                    new_loc = (agent.get_x() - 1, agent.get_y())
                    if self.is_open_tile(new_loc):
                        self.move_agent_to(agent, new_loc)
            elif thought == Actions.MOVE_FORWARD:
                dir = agent.get_forward()
                if dir == Directions.UP:
                    new_loc = (agent.get_x(), agent.get_y() - 1)
                    if self.is_open_tile(new_loc):
                        self.move_agent_to(agent, new_loc)
                elif dir == Directions.DOWN:
                    new_loc = (agent.get_x(), agent.get_y() + 1)
                    if self.is_open_tile(new_loc):
                        self.move_agent_to(agent, new_loc)
                elif dir == Directions.RIGHT:
                    new_loc = (agent.get_x() + 1, agent.get_y())
                    if self.is_open_tile(new_loc):
                        self.move_agent_to(agent, new_loc)
                elif dir == Directions.LEFT:
                    new_loc = (agent.get_x() - 1, agent.get_y())
                    if self.is_open_tile(new_loc):
                        self.move_agent_to(agent, new_loc)
            elif thought == Actions.MOVE_BACKWARD:
                dir = agent.get_forward()
                if dir == Directions.UP:
                    new_loc = (agent.get_x(), agent.get_y() + 1)
                    if self.is_open_tile(new_loc):
                        self.move_agent_to(agent, new_loc)
                elif dir == Directions.DOWN:
                    new_loc = (agent.get_x(), agent.get_y() - 1)
                    if self.is_open_tile(new_loc):
                        self.move_agent_to(agent, new_loc)
                elif dir == Directions.RIGHT:
                    new_loc = (agent.get_x() - 1, agent.get_y())
                    if self.is_open_tile(new_loc):
                        self.move_agent_to(agent, new_loc)
                elif dir == Directions.LEFT:
                    new_loc = (agent.get_x() + 1, agent.get_y())
                    if self.is_open_tile(new_loc):
                        self.move_agent_to(agent, new_loc)
            elif thought == Actions.STAND_STILL:
                pass
            elif thought == Actions.RELEASE_PHEROMONE:
                if World.PHEROMONES_ENABLED:
                    self.pheromones.append((agent.get_loc(), sum(agent.brain.genome), self.cur_step))
            else:
                assert False, "did you forget to add an action?"
                
        self.cur_step += 1
        
        #print(f"step took {(time.time() - cur) * 1000}ms")
        
        # remove old pheromones
        if World.PHEROMONES_ENABLED:
            for pheromone in self.pheromones:
                _, _, step_created = pheromone
                if self.cur_step - step_created > World.PHEROMONE_STEP_DURATION:
                    self.pheromones.remove(pheromone)
                else:
                    break
        
        return self.cur_step


pygame.init()


screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])

running = True
clock = pygame.time.Clock()

cur_time: float = 0
last_time: float = 0
delta_time: float = 0
counter: int = 0
MAX_COUNTER = 1
DEBUG_DISPLAY_FPS = False

# MODEL PARAMETERS
INITIAL_POP_SIZE = 250
STEPS_PER_GEN = 250

world = World.create_new_world(128, 128, INITIAL_POP_SIZE)

# TODO serialize world to file each generation, so work can be done without pygame, and done after to show specific generations

running = True

while running:
    clock.tick(144)
    
    cur_time += clock.get_rawtime()
    delta_time = (cur_time - last_time) / 1000
    if delta_time == 0:
        delta_time = .999
    
    #display fps?
    if DEBUG_DISPLAY_FPS:
        if counter >= MAX_COUNTER:
            print(f"FPS: {1 / delta_time}")
            counter -= MAX_COUNTER
        else:
            counter += 1 * delta_time
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    step = world.step()
    
    if step >= STEPS_PER_GEN:
            
        did_surive_func = lambda agent: agent.get_x() > (world.width * 5 / 8) and agent.get_y() >= (world.height/3) and agent.get_y() <= (world.height*2) / 3
        new_agents = World.create_new_generation(world.agents, did_surive_func, max_pop_size=INITIAL_POP_SIZE)
        world.reset()
        #world.create_wall(75, 5, 33, 60)
        world.set_agents(new_agents)
        #print(len(world.agents))
        print(f"generation: {world.cur_gen}")
    
    # Fill the background with white
    screen.fill((255, 255, 255))
    
    
    world.draw(screen)
    
    pygame.display.flip()
    
    last_time = cur_time
    
pygame.quit()