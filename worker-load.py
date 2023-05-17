# %%

import IPython
from matplotlib.axes import Axes
import agentpy as ap
import numpy as np
import matplotlib.pyplot as plt

# %%


class WorkerAgent(ap.Agent):
    def setup(self):
        # Initialize an attribute with a parameter
        self.processing_capacity = self.p.processing_capacity
        self.max_demand = int(self.p.max_demand_factor * self.processing_capacity)
        self.current_load = 0
        self.added_load = 0
        self.is_blocking = False

    def add_load(self):
        self.added_load = self.get_random_load()

        self.current_load = max(
            self.current_load + self.added_load - self.processing_capacity, 0
        )

        self.is_blocking = self.current_load > self.processing_capacity

    def get_random_load(self):
        raw = sorted(np.random.exponential(0.42, self.max_demand))
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

        self.load_average = self.total_load / self.p.actor_count
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
        self.record("load_average")
        self.record("open_capacity")

    def end(self):
        # Called at the end of the simulation
        self.report("open_capacity")  # Report a simulation result


run_parameters = {
    "actor_count": 4,
    "steps": 50,
    "processing_capacity": 10,
    "max_demand_factor": 2,
}


model = WorkerModel(parameters=run_parameters)

results = model.run()

df = results.variables.WorkerModel
df.drop(index=0, inplace=True)
# %%
# Inspect

df.describe()

# %%
# Visualize flatly

title = f'{run_parameters["actor_count"]} actors with capacity {run_parameters["processing_capacity"]}\n {run_parameters["max_demand_factor"]} demand limit'

for idx, column in enumerate(df.columns):
    plt.scatter(df.index, df[column], marker=".", color="g", s=1, lw=1)
    plt.legend([column])
    plt.title(title)
    plt.show()


# %% 
# Visualize Animated 


def animation_plot(model, ax: Axes):
    ax.set_title(f"Average Load & Blocking Count")
    ax.set_xlim((0, model.p.steps))
    ax.set_ylim((0, max(model.p.actor_count, model.p.processing_capacity)))

    ax.plot(model.log["t"], model.log["blocking_count"])
    ax.plot(model.log["t"], model.log["load_average"], label="load_average")
    
    ax.legend(["load_average", "blocking_count"])


fig, ax = plt.subplots()
model = WorkerModel(run_parameters)
animation = ap.animate(model, fig, ax, animation_plot)
IPython.display.HTML(animation.to_jshtml(fps=15))
# %%
