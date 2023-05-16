# %%
import agentpy as ap
import numpy as np
import matplotlib.pyplot as plt

# %%


class WorkerAgent(ap.Agent):
    def setup(self):
        # Initialize an attribute with a parameter
        self.processing_capacity = self.p.processing_capacity
        self.max_demand = self.p.max_demand
        self.current_load = 0
        self.added_load = 0
        self.is_blocking = False

    def add_load(self):
        self.added_load = self.get_random_load()

        self.current_load = max(
            self.current_load + self.added_load -self.processing_capacity, 0
        )

        self.is_blocking = self.current_load > self.processing_capacity

    def get_random_load(self):
        raw = sorted(np.random.exponential(0.42,self.max_demand))
        y = raw / max(raw)
        return self.model.nprandom.choice(y) * self.max_demand


class WorkerModel(ap.Model):
    def setup(self):
        # Called at the start of the simulation
        self.agents = ap.AgentList(self, self.p.actor_count, WorkerAgent)

    def step(self):
        # Called at every simulation step
        self.agents.add_load()  # Call a method for every agent

    def update(self):
        # Called after setup as well as after each step
        self.blocking_count = len([True for agent in self.agents if agent.is_blocking])

        self.total_load = sum([a.current_load for a in self.agents])
        self.total_added_load = sum([a.current_load for a in self.agents])

        self.open_capacity = sum(
            [
                self.p.processing_capacity - a.current_load
                for a in self.agents
                if not a.is_blocking
            ]
        )

        self.record("blocking_count")
        self.record("total_load")
        self.record("total_added_load")
        self.record("open_capacity")

    def end(self):
        # Called at the end of the simulation
        self.report("open_capacity")  # Report a simulation result


run_parameters = {
    "actor_count": 16,
    "steps": 100,
    "processing_capacity": 10,
    "max_demand": 15,
}


model = WorkerModel(parameters=run_parameters)

results = model.run()

df = results.variables.WorkerModel

#%%
# Visualize

title = f'{run_parameters["actor_count"]} actors with capacity {run_parameters["processing_capacity"]}\n {run_parameters["max_demand"]} demand limit'

for idx, column in enumerate(df.columns):
    plt.scatter(df.index, df[column], marker='.', )    
    plt.legend([column])
    plt.title(title)
    plt.show()
    


#%%
# Inspect
df[column].describe()
df[column].value_counts()

