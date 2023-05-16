# %%
import agentpy as ap

import matplotlib.pyplot as plt


# %% The Agent
class MatchAgent(ap.Agent):
    def setup(self):
        # Initialize an attribute with a parameter
        self.selectivity = self.p.selectivity
        self.interest_ids = set()
        self.matched = False

    def gather_interest(self):
        self.interest_ids = self.p.strategy(self)

    def check_match(self, other):
        if (
            self.id != other.id
            and self.id in other.interest_ids
            and other.id in self.interest_ids
        ):
            return True

        return False

    def anyone_strategy(agent):
        interest_count = int(agent.p.actor_count * agent.selectivity / 100)
        results = {a.id for a in agent.model.agents.random(n=interest_count)}
        results.discard(agent.id)
        return results

    def opposites_strategy(agent):
        opposites = range(agent.id % 2, agent.p.actor_count, 2)
        interest_count = int(len(opposites) * agent.selectivity / 100)
        result = {
            a.id for a in agent.model.agents.select(opposites).random(n=interest_count)
        }
        result.discard(agent.id)
        return result


class MatchingModel(ap.Model):
    def setup(self):
        # Called at the start of the simulation
        self.agents = ap.AgentList(self, self.p.actor_count, MatchAgent)

    def step(self):
        # Called at every simulation step
        self.agents.gather_interest()  # Call a method for every agent

    def update(self):
        # Called after setup as well as after each step
        self.match_count = 0
        self.matches = []
        all = set(self.agents)
        while len(all) >= 2:
            a = all.pop()
            for b in all:
                if a.check_match(b):
                    self.match_count += 1
                    # cross-remove interest, already matched
                    a.interest_ids.clear()
                    b.interest_ids.clear()
                    self.matches.append(set([a.id, b.id]))
                    break

        self.record("match_count")
        self.record("matches")

    def end(self):
        # Called at the end of the simulation
        self.report("match_count")  # Report a simulation result


# %% The execution

run_parameters = {
    "steps": 1000,
    "actor_count": 400,
    "selectivity": 2,
    "strategy": MatchAgent.opposites_strategy,
}

model = MatchingModel(parameters=run_parameters)

results = model.run()

# first run always empty in our case
results.variables.MatchingModel.drop(index=0, inplace=True)


# Visualize
title = f'{run_parameters["selectivity"]}% pickyness\n{run_parameters["actor_count"]} actors using {run_parameters["strategy"].__name__} \nX {run_parameters["steps"]} times'
var = 'match_count'
plt.hist(results.variables.MatchingModel[var])
plt.title(title)
plt.legend([var])


# %% The Results
results.info


# %%
results.variables.MatchingModel
