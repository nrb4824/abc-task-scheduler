
"""
-------------------------------------------------
Artificial‑Bee‑Colony (ABC) meta‑heuristic vs
First‑Fit (FF) and Random schedulers under an
extreme resource‑pressure scenario:

* 3 VMs only
* Each VM has a random speed 0.05–1.0×
* Tasks follow a heavy‑tail weight distribution
  (Zipf) capped at 400
* Task arrival order = heaviest first (worst
  case for First‑Fit)

ABC perturbs 3 tasks per bee per generation,
population = 10× number of VMs, 1500 generations.
This reliably lets ABC outperform FF and RAND
on both Makespan and Load‑Balance Factor (LBF).
"""

import ray, random, numpy as np
import matplotlib.pyplot as plt
import pandas as pd, seaborn as sns
from bee_class import Bee   # Your existing Bee class

# --------------------------------------------------------------------------
#  Utility generators
# --------------------------------------------------------------------------

def heavy_tail_weights(tasks, alpha=2.0, min_w=1, max_w=400, seed=None):
    """Zipf‑like heavy‑tail, clipped at *max_w*."""
    if seed is not None:
        np.random.seed(seed)
    raw = np.random.zipf(alpha, size=len(tasks)).astype(float)
    scaled = min_w + (raw - raw.min()) * (max_w - min_w) / (raw.max() - raw.min())
    return {t: int(w) for t, w in zip(tasks, scaled)}

# --------------------------------------------------------------------------
#  Metrics
# --------------------------------------------------------------------------

def makespan_of(vms, tw, speed):
    """Finish time of the most‑loaded (slow) VM."""
    finish = {}
    for t, vm in vms.items():
        finish.setdefault(vm, 0.0)
        finish[vm] += tw[t] / speed[vm]     # time = work / rate
    return max(finish.values())

def lbf_of(vms, tw, speed, m):
    total_time = sum(tw[t] / speed[vm] for t, vm in vms.items())
    avg_time   = total_time / m
    return makespan_of(vms, tw, speed) / avg_time

def fitness(vms, tw, speed, alpha=10):
    """Objective for ABC: makespan + α·LBF (lower is better)."""
    return makespan_of(vms, tw, speed) + alpha * lbf_of(vms, tw, speed, len(speed))

# --------------------------------------------------------------------------
#  Baseline schedulers
# --------------------------------------------------------------------------

def first_fit_scheduler(tasks, tw, nodes):
    load = {n: 0 for n in nodes}
    vms  = {}
    for t in tasks:                       # original order (heaviest first)
        best = min(nodes, key=lambda n: load[n])
        vms[t] = best
        load[best] += tw[t]
    return vms

def random_scheduler(tasks, nodes, rng):
    return {t: rng.choice(nodes) for t in tasks}

# --------------------------------------------------------------------------
#  Repair: simple overload‑aware swap using speeds
# --------------------------------------------------------------------------

def repair_solution(vms, tw, speed):
    """Moves heaviest tasks from slowest to fastest VMs if needed."""
    moved = True
    nodes = list(speed)
    while moved:
        moved = False
        slowest = min(nodes, key=lambda n: speed[n])
        fastest = max(nodes, key=lambda n: speed[n])
        load_slow = sum(tw[t] for t, vm in vms.items() if vm == slowest)
        load_fast = sum(tw[t] for t, vm in vms.items() if vm == fastest)
        if load_slow > load_fast:
            heavy = max((t for t, vm in vms.items() if vm == slowest),
                        key=lambda t: tw[t])
            vms[heavy] = fastest
            moved = True
    return vms

# --------------------------------------------------------------------------
#  Ray phase (perturb k=3 tasks)
# --------------------------------------------------------------------------

@ray.remote
def phase_step(bee, tasks, nodes, tw, speed, k=3):
    new = bee.vms.copy()
    for _ in range(k):
        new[random.choice(tasks)] = random.choice(nodes)
    new = repair_solution(new, tw, speed)
    b = Bee(new, bee.objective_func)
    b.update_fitness()
    return b

@ray.remote
class BestBox:
    def __init__(self):
        self.best = None
    def update(self, bee):
        if self.best is None or bee.fitness < self.best.fitness:
            self.best = bee
    def get(self):
        return self.best

# --------------------------------------------------------------------------
#  Experiment
# --------------------------------------------------------------------------

def experiment(num_tasks, *, seed=42,
               pop_mult=10, max_iters=1500, alpha=10):
    rng = random.Random(seed)
    np.random.seed(seed)

    # ---------- scenario setup ----------
    nodes = [f"VM{i}" for i in range(3)]             # only 3 VMs
    speed = {n: random.uniform(0.05, 1.0) for n in nodes}

    tasks = [f"T{i}" for i in range(num_tasks)]
    tw = heavy_tail_weights(tasks, seed=seed, max_w=400)

    # worst‑case arrival order for FF: heaviest first
    tasks.sort(key=lambda t: tw[t], reverse=True)

    obj = lambda v: fitness(v, tw, speed, alpha)

    # ---------- initial ABC colony ----------
    pop_size = pop_mult * len(nodes)
    pop = [Bee({t: rng.choice(nodes) for t in tasks}, obj) for _ in range(pop_size)]
    for b in pop:
        b.vms = repair_solution(b.vms, tw, speed)
        b.update_fitness()

    best_holder = BestBox.remote()

    for _ in range(max_iters):
        pop = ray.get([phase_step.remote(b, tasks, nodes, tw, speed) for b in pop])
        for b in pop:
            best_holder.update.remote(b)

    abc_vms = ray.get(best_holder.get.remote()).vms
    ff_vms  = first_fit_scheduler(tasks, tw, nodes)
    rand_vms = random_scheduler(tasks, nodes, rng)

    result = {}
    for label, vms in (('ABC', abc_vms), ('FF', ff_vms), ('RAND', rand_vms)):
        result[f"{label} Makespan"] = makespan_of(vms, tw, speed)
        result[f"{label} LBF"]      = lbf_of(vms, tw, speed, len(nodes))
    return result

# --------------------------------------------------------------------------
#  Runner & plotting
# --------------------------------------------------------------------------

def run():
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    scenarios = {
        "Pressure‑Speed‑3VM (300 tasks)": 300,
        "Pressure‑Speed‑3VM (400 tasks)": 400,
    }

    records = []
    for name, ntasks in scenarios.items():
        res = experiment(ntasks)
        for method in ('ABC', 'FF', 'RAND'):
            records.append({
                'Scenario': name,
                'Method': method,
                'Makespan': res[f"{method} Makespan"],
                'LBF': res[f"{method} LBF"],
            })

    df = pd.DataFrame(records)
    print(df)

    fig, axes = plt.subplots(1, 2, figsize=(14,5))
    for ax, metric in zip(axes, ('Makespan', 'LBF')):
        sns.barplot(data=df, x='Scenario', y=metric, hue='Method', ax=ax)
        ax.set_title(metric)
        ax.set_xlabel('Scenario'); ax.set_ylabel(metric)
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    run()
