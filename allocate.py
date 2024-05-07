# Import the Gurobi functions and classes
import gurobipy as gp
from gurobipy import GRB, Var
from typing import List
import random
import numpy as np

class Aqua_allocations():
    def __init__(self, machines: int, gpus_per_machine: int, models: int, model_memories: List[int], model_names: List[str], gpu_memory: int) -> None:
        self.num_machines: int = machines
        self.gpus_per_machine: int = gpus_per_machine
        self.models: int = models
        self.model_locations: List[List[Var]] = []
        self.model = gp.Model("aqua")
        self.model_memories = model_memories
        self.objective_variable: Var = None
        self.model_names = model_names
        self.memory_per_gpu = gpu_memory
    
    def add_model_location_variables(self):
        for m in range(self.models):
            model_var_list = []
            for s in range(self.num_machines):
                model_location_var = self.model.addVar(vtype=GRB.BINARY, name='is_model_{}_on_{}'.format(m, s))
                model_var_list.append(model_location_var)
            self.model_locations.append(model_var_list)
    
    def add_one_gpu_per_model_constraint(self):
        for model_location in self.model_locations:
            self.model.addConstr(gp.quicksum(model_location) <= 1)
            self.model.addConstr(gp.quicksum(model_location) >= 1)

    def add_one_model_per_gpu_constraint(self):
        for machine in range(self.num_machines):
            models_on_machines = [locations[machine] for locations in self.model_locations]
            self.model.addConstr(gp.quicksum(models_on_machines) <= self.gpus_per_machine)
    
    def get_lower_negative_bound(self):
        max_negative = 0
        for memory in self.model_memories:
            if memory < 0:
                max_negative += memory
        return max_negative

    def get_lb_for_machine_memory(self):
        return self.gpus_per_machine * min(self.model_memories)
    
    def get_lb_for_machine_models(self):
        return -1 * self.gpus_per_machine * self.memory_per_gpu

    def set_objective(self):
        model_memories = [abs(memory) for memory in self.model_memories]
        is_model_producer = [1 if memory > 0 else -1 for memory in self.model_memories]

        memory_per_machine = []
        producers_consumers_per_machine = []

        for machine in range(self.num_machines):
            machine_memory = [locations[machine] * is_model_producer[model] * model_memories[model] for model, locations in enumerate(self.model_locations)]
            machine_memory_var = self.model.addVar(vtype=GRB.CONTINUOUS, name='machine_memory_var_{}'.format(machine), lb=self.get_lb_for_machine_memory())
            self.model.addConstr(machine_memory_var >= gp.quicksum(machine_memory))
            self.model.addConstr(machine_memory_var <= gp.quicksum(machine_memory))

            machine_models = [locations[machine] * is_model_producer[model] * self.memory_per_gpu for model, locations in enumerate(self.model_locations)]
            machine_models_var = self.model.addVar(vtype=GRB.CONTINUOUS, name='machine_models_var_{}'.format(machine), lb=self.get_lb_for_machine_models())
            self.model.addConstr(machine_models_var >= gp.quicksum(machine_models))
            self.model.addConstr(machine_models_var <= gp.quicksum(machine_models))

            memory_per_machine.append(machine_memory_var)
            producers_consumers_per_machine.append(machine_models_var)
        
        max_memory = self.model.addVar(vtype=GRB.CONTINUOUS, name='max_memory')
        max_producers_consumers = self.model.addVar(vtype=GRB.CONTINUOUS, name='max_producers_consumers')

        self.model.addConstr(max_memory == gp.max_(memory_per_machine))
        self.model.addConstr(max_producers_consumers == gp.max_(producers_consumers_per_machine))

        self.model.setObjective(max_memory + max_producers_consumers, sense=GRB.MINIMIZE)

    def optimize(self):
        self.model.optimize()
        server_to_producer_consumer = {}
        server_models = {}

        for model, locations in enumerate(self.model_locations):
            for server, server_var in enumerate(locations):
                if server_var.X > 0.9:
                    if server not in server_to_producer_consumer:
                        server_to_producer_consumer[server] = [0, 0]
                        server_models[server] = []
                    server_models[server].append((self.model_names[model], self.model_memories[model]))

                    if self.model_memories[model] > 0:
                        server_to_producer_consumer[server][0] += 1
                    else:
                        server_to_producer_consumer[server][1] += 1
        
        print(server_to_producer_consumer)

        for s in range(self.num_machines):
            mem_var = self.model.getVarByName("machine_memory_var_{}".format(s))
            memory_usage = int(mem_var.X)
            print('Machine {}, memory: {}, models: {}, var: {}'.format(s, memory_usage, server_models[s], mem_var.VarName))
            model_names = [model_name for model_name, _ in server_models[s]]
            print('EXEC: {}'.format(','.join(model_names)))

def sample_models(model_names_list, model_memories_list, need_to_sample, destination_names, destination_memories):
    models = [(model_name, model_memory) for (model_name, model_memory) in zip(model_names_list, model_memories_list)]
    sampling_list = [m for m in models]
    print('Sampling {} from: {}'.format(need_to_sample, models))
    num_sampled = 0
    while num_sampled < need_to_sample:
        choice = np.random.choice([m for m in range(len(sampling_list))])
        model_name, model_memory = sampling_list[choice]
        sampling_list.remove(sampling_list[choice])
        destination_names.append(model_name)
        destination_memories.append(model_memory)
        num_sampled += 1
        if len(sampling_list) == 0:
            sampling_list = [m for m in models]


def configure_cluster(audio_fraction, vision_fraction, llm_donor_fraction, total_gpus):
    donor_vision = ['sd', 'sdxl', 'kand']
    donor_vision_memories = [30, 20, 40]

    donor_audio = ['audio', 'music']
    donor_audio_memories = [35, 15]

    donor_llms = ['mistral', 'llama']
    donor_llms_memories = [45, 40]

    # d_ids = [r for r in range(len(donor_model_list))]

    receiver_model_names = ['opt', 'lora', 'cfs']
    receiver_model_list = [-20, -10, -30]

    num_audio_models = int(audio_fraction * total_gpus)
    num_vision_models = int(vision_fraction * total_gpus)
    num_llm_donors = int(llm_donor_fraction * total_gpus)
    donor_gpus = (num_audio_models + num_vision_models + num_llm_donors)
    receiver_gpus = total_gpus - donor_gpus


    print('After configuration, number of donors: {}, receivers: {}'.format(donor_gpus, receiver_gpus))

    model_memories = []
    model_names = []

    sample_models(donor_vision, donor_vision_memories, num_vision_models, model_names, model_memories)
    sample_models(donor_audio, donor_audio_memories, num_audio_models, model_names, model_memories)
    sample_models(donor_llms, donor_llms_memories, num_llm_donors, model_names, model_memories)
    sample_models(receiver_model_names, receiver_model_list, receiver_gpus, model_names, model_memories)

    return model_memories, model_names

def solve_placement(num_machines, num_gpus_per_machine, total_gpus, model_memory_requirements, model_names, gpu_memory):
    allocations = Aqua_allocations(num_machines, num_gpus_per_machine, total_gpus, model_memory_requirements, model_names, gpu_memory)
    allocations.add_model_location_variables()
    allocations.add_one_gpu_per_model_constraint()
    allocations.add_one_model_per_gpu_constraint()
    allocations.set_objective()
    allocations.optimize()

audio_fraction = 1/3.0 * 0
vision_fraction = 1/3.0 * 0
llm_donor_fraction = 0.5

for num_machines in [2, 4, 6, 8, 12, 16, 32]:
    num_gpus_per_machine = 8
    np.random.seed(4)

    total_gpus = num_machines * num_gpus_per_machine

    model_memory_requirements, model_names = configure_cluster(audio_fraction=audio_fraction, vision_fraction=vision_fraction, llm_donor_fraction=llm_donor_fraction, total_gpus=total_gpus)

    solve_placement(num_machines, num_gpus_per_machine, total_gpus, model_memory_requirements, model_names, 80)
    

