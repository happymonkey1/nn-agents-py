from bitstring import BitArray
import numpy as np
import random

# setup properties
WEIGHT_FACTOR = 8200
GENOME_SIZE = 24
NUM_INPUT = 4
NUM_HIDDEN = 4
NUM_OUTPUT = 2

def set_genome_size(new_size: int):
    global GENOME_SIZE
    GENOME_SIZE = new_size
    
def set_weight_factor(new_weight_factor: int):
    global WEIGHT_FACTOR
    WEIGHT_FACTOR = new_weight_factor
    
def set_num_input(new_input: int):
    global NUM_INPUT
    NUM_INPUT = new_input
    
def set_num_hidden(new_hidden: int):
    global NUM_HIDDEN
    NUM_HIDDEN = new_hidden
    
def set_num_output(new_output: int):
    global NUM_OUTPUT
    NUM_OUTPUT = new_output

class Gene:
    
    INPUT_ID = 0
    HIDDEN_ID = 1
    OUTPUT_ID = 0
    
    def __init__(self, src_type, src_id, sink_type, sink_id, weight):
        self.source_type = src_type
        self.source_id = src_id
        self.sink_type = sink_type
        self.sink_id = sink_id
        self.weight = weight
        
    def __str__(self):
        gene_encoding = Gene.encode(self.source_type, self.source_id, self.sink_type, self.sink_id, self.weight)
        return f"""{hex(gene_encoding)}\n  source type: '{self.source_type}'\n  source id: '{bin(self.source_id)}'\n  sink type: '{self.sink_type}'\n  sink id: '{bin(self.sink_id)}'\n  weight: '{bin(self.weight)}'"""
    
    def __repr__(self):
        return f"""Gene(src_type={self.source_type}, src_id={bin(self.source_id)}, sink_type={self.sink_type}, sink_id={bin(self.sink_id)}, weight={bin(self.weight)})"""
    
    @staticmethod
    def create_random_gene():
        return random.getrandbits(32)
    
    @staticmethod
    def create_random_genome(genome_size: int) -> list[int]:
        return [Gene.create_random_gene() for _ in range(genome_size)]
        
    @staticmethod
    def mutate(gene):
        # TODO: this is very slow
        g = f"{gene:032b}"
        bit_flip_index = random.randint(0,31)
        flip = "0" if g[bit_flip_index] == "1" else "1"
        
        g = g[:bit_flip_index] + flip + g[bit_flip_index + 1:]
        return int(g, 2)
    
    @staticmethod
    def encode(src_type, src_id, sink_type, sink_id, weight):
        encoding = src_type
        encoding = (encoding << 7)  | src_id
        encoding = (encoding << 1)  | sink_type
        encoding = (encoding << 7)  | sink_id
        encoding = (encoding << 16) | weight

        return encoding
    
    @staticmethod
    def decode(gene):
        weight = gene & 0xFFFF
        sink_id = (gene >> 16) & 0b1111111
        sink_type = (gene >> 23) & 1
        source_id = (gene >> 24) & 0b1111111
        source_type = (gene >> 31) & 1
        
        return Gene(source_type, source_id, sink_type, sink_id, weight)   

class ActivationFunctions:
    TANH = lambda x: (np.exp(x, dtype=np.float32) - np.exp(-x, dtype=np.float32)) / (np.exp(x, dtype=np.float32) + np.exp(-x, dtype=np.float32))
    SIGMOID = lambda x: 1 / (1 + np.exp(-x, dtype=np.float32))

class Network:
    def __init__(self, num_input: int, num_hidden: int, num_output:int, has_bias: bool = True):
        self.num_input = num_input + (1 if has_bias else 0)
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.hidden_activation = ActivationFunctions.TANH
        self.output_activation = ActivationFunctions.TANH
        self.has_bias = has_bias
        
        # input --> hidden
        weights0 = np.zeros((self.num_hidden, self.num_input), dtype=np.float32)
        # hidden --> output
        weights1 = np.zeros((self.num_output, self.num_hidden), dtype=np.float32)
        # hidden --> hidden
        weights2 = np.zeros((self.num_hidden, self.num_hidden), dtype=np.float32)
        # input --> output
        weights3 = np.zeros((self.num_output, self.num_input), dtype=np.float32)
        
        self.weight_layers = [weights0, weights1, weights2, weights3]
        
    def set_genome(self, genome: list[int]):
        self.genome = genome
    
    def get_weights_input_to_hidden(self):
        return self.weight_layers[0]
    
    def get_weights_hidden_to_output(self):
        return self.weight_layers[1]
    
    def get_weights_hidden_to_hidden(self):
        return self.weight_layers[2]
    
    def get_weights_input_to_output(self):
        return self.weight_layers[3]
    
    def get_brain_as_str(self) -> str:
        genome_str = f"genome: {' '.join([hex(gene) for gene in self.genome])}"
        
        weights = ""
        i: int = 0
        for weight_layer in self.weight_layers:
            weights += f"weights{i}: {weight_layer}\n"
            i += 1
            
        return genome_str + "\n" + weights
    
    def feed_forward(self, inputs):
        assert inputs.shape[0] == self.num_input
        #X = np.array(inputs)
        #if self.has_bias:
        #   X = np.append(X, [1.0])
        
        # input --> hidden + activation
        res1 = self.hidden_activation(self.weight_layers[0].dot(inputs))
       # print(f"input-->hidden: {res1}")
        
        # hidden --> hidden + activation
        res2 = self.hidden_activation(self.weight_layers[2].dot(res1))
        #print(f"hidden-->hidden: {res2}")
        
        # hidden --> output
        res3 = self.weight_layers[1].dot(res2)
        #print(f"hidden-->output: {res3}")
        # input --> output
        res4 = self.weight_layers[3].dot(inputs)
        #print(f"input-->output: {res4}")
        
        # output sum + activation
        return self.output_activation(res3 + res4)
    
    @staticmethod
    def from_genome(genome):
        
        network = Network(num_input=NUM_INPUT, num_hidden=NUM_HIDDEN, num_output=NUM_OUTPUT, has_bias=True)
        network.set_genome(genome)
        
        for gene_encoded in genome:
            gene = Gene.decode(gene_encoded)
            
            source_index = None
            sink_index = None
            layer_index = None
            # input --> output
            if gene.source_type == Gene.INPUT_ID and gene.sink_type == Gene.OUTPUT_ID:
                source_index = gene.source_id % network.num_input
                sink_index = gene.sink_id % network.num_output
                layer_index = 3
            # input --> hidden
            elif gene.source_type == Gene.INPUT_ID and gene.sink_type == Gene.HIDDEN_ID:
                source_index = gene.source_id % network.num_input
                sink_index = gene.sink_id % network.num_hidden
                layer_index = 0
            # hidden --> output
            elif gene.source_type == Gene.HIDDEN_ID and gene.sink_type == Gene.OUTPUT_ID:
                source_index = gene.source_id % network.num_hidden
                sink_index = gene.sink_id % network.num_output
                layer_index = 1
            # hidden --> hidden
            elif gene.source_type == Gene.HIDDEN_ID and gene.sink_type == Gene.HIDDEN_ID:
                source_index = gene.source_id % network.num_hidden
                sink_index = gene.sink_id % network.num_hidden
                layer_index = 2
            else:
                assert False, f"invalid combination of source_id='{gene.source_type}' and sink_id='{gene.sink_type}'"
            
            assert source_index is not None, "invalid source neuron index"
            assert sink_index is not None, "invalid sink neuron index" 
            assert layer_index is not None, "invalid layer index"
            
            weight = BitArray(bin=f"{gene.weight:016b}").int / WEIGHT_FACTOR
            network.weight_layers[layer_index][sink_index][source_index] = weight
        
        return network