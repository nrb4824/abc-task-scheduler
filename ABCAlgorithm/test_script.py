import ray
import random
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from bee_class import Bee  # your existing Bee class

# Initialize Ray
ray.init(ignore_reinit_error=True)

def generate_random_solution(tasks, nodes):
    return {t: random.choice(nodes) for t in tasks}

def evaluate_fitness(vms, task_weights, node_thresholds):
    load = {n: 0 for n in node_thresholds}
    for t, n in vms.items():
        load[n] += task_weights[t]
    fitness = 0
    for n, w in load.items():
        thr = node_thresholds[n]
        if w > thr:
            fitness += 1000 * (w - thr)
        else:
            fitness += (thr - w)
    return fitness

def repair_solution(vms, task_weights, node_thresholds):
    node_load = {n: 0 for n in node_thresholds}
    for t, n in vms.items():
        node_load[n] += task_weights[t]
    for n in node_thresholds:
        while node_load[n] > node_thresholds[n]:
            assigned = [t for t, vm in vms.items() if vm == n]
            assigned.sort(key=lambda t: task_weights[t], reverse=True)
            moved = False
            for t in assigned:
                for alt in node_thresholds:
                    if alt == n:
                        continue
                    alt_load = sum(task_weights[x] for x, vm in vms.items() if vm == alt)
                    if alt_load + task_weights[t] <= node_thresholds[alt]:
                        vms[t] = alt
                        node_load[n] -= task_weights[t]
                        node_load[alt] += task_weights[t]
                        moved = True
                        break
                if moved:
                    break
            if not moved:
                break
    return vms

@ray.remote
def employed_phase(bee, tasks, nodes, tw, nt):
    new = bee.vms.copy()
    new[random.choice(tasks)] = random.choice(nodes)
    new = repair_solution(new, tw, nt)
    b = Bee(new, bee.objective_func)
    b.update_fitness()
    return b

@ray.remote
def onlooker_phase(bee, tasks, nodes, tw, nt):
    new = bee.vms.copy()
    if random.random() < 0.5:
        new[random.choice(tasks)] = random.choice(nodes)
    new = repair_solution(new, tw, nt)
    b = Bee(new, bee.objective_func)
    b.update_fitness()
    return b

@ray.remote
def scout_phase(tasks, nodes, tw, nt, obj):
    new = generate_random_solution(tasks, nodes)
    new = repair_solution(new, tw, nt)
    b = Bee(new, obj)
    b.update_fitness()
    return b

@ray.remote
class BestSolution:
    def __init__(self):
        self.best = None
    def update(self, bee):
        if self.best is None or bee.fitness < self.best.fitness:
            self.best = bee
    def get(self):
        return self.best


def list_scheduler(tasks, tw, nodes):
    load = {n: 0 for n in nodes}
    vms = {}
    for t in sorted(tasks, key=lambda x: tw[x], reverse=True):
        best_vm = min(nodes, key=lambda n: load[n])
        vms[t] = best_vm
        load[best_vm] += tw[t]
    return vms, max(load.values())


def makespan_of(vms, tw):
    load = {}
    for t, n in vms.items():
        load.setdefault(n, 0)
        load[n] += tw[t]
    return max(load.values())

def lbf_of(vms, tw, num_vms):
    total = sum(tw.values())
    avg = total / num_vms
    mk = makespan_of(vms, tw)
    return mk / avg


def experiment(num_tasks, num_vms, max_iters=30):
    tasks = [f"task{i}" for i in range(1, num_tasks + 1)]
    nodes = [f"vm{i}" for i in range(1, num_vms + 1)]
    tw = {t: random.randint(1, 10) for t in tasks}
    total_weight = sum(tw.values())
    avg_load = total_weight / num_vms
    nt = {n: random.randint(int(avg_load * 0.8), int(avg_load * 1.2)) for n in nodes}

    obj = lambda v: evaluate_fitness(v, tw, nt)
    pop = [Bee(generate_random_solution(tasks, nodes), obj) for _ in range(num_vms)]
    for b in pop:
        b.vms = repair_solution(b.vms, tw, nt)
        b.update_fitness()

    best_actor = BestSolution.remote()
    start_abc = time.time()
    for _ in range(max_iters):
        eb = ray.get([employed_phase.remote(b, tasks, nodes, tw, nt) for b in pop])
        ob = ray.get([onlooker_phase.remote(b, tasks, nodes, tw, nt) for b in eb])
        for b in ob:
            best_actor.update.remote(b)
        stagnants = [i for i, b in enumerate(ob) if b.fitness > 50]
        if stagnants:
            sc = ray.get([scout_phase.remote(tasks, nodes, tw, nt, obj) for _ in stagnants])
            for i, b_new in zip(stagnants, sc):
                pop[i] = b_new
        else:
            pop = ob
    abc_time = time.time() - start_abc
    best_bee = ray.get(best_actor.get.remote())
    abc_mk = makespan_of(best_bee.vms, tw)
    abc_lbf = lbf_of(best_bee.vms, tw, num_vms)

    start_gen = time.time()
    gen_vms, gen_mk = list_scheduler(tasks, tw, nodes)
    gen_time = time.time() - start_gen
    gen_lbf = lbf_of(gen_vms, tw, num_vms)

    return {
        "ABC Makespan": abc_mk,
        "ABC Time": abc_time,
        "ABC LBF": abc_lbf,
        "ABC VMS": best_bee.vms,
        "ABC TW": tw,
        "GEN Makespan": gen_mk,
        "GEN Time": gen_time,
        "GEN LBF": gen_lbf,
        "GEN VMS": gen_vms,
        "GEN TW": tw
    }


def run_all_scenarios(scenarios, max_iters=30):
    records = []
    distribution_data = {}
    for name, (ntasks, nvms) in scenarios.items():
        res = experiment(ntasks, nvms, max_iters)
        for method in ["ABC", "GEN"]:
            records.append({
                "Scenario": name,
                "Method": method,
                "Makespan": res[f"{method} Makespan"],
                "LBF": res[f"{method} LBF"]
            })
            dist = {}
            for t, vm in res[f"{method} VMS"].items():
                dist[vm] = dist.get(vm, 0) + res[f"{method} TW"][t]
            distribution_data[(name, method)] = dist
    return pd.DataFrame(records), distribution_data


def plot_multiscenario(df):
    metrics = ["Makespan", "LBF"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, metric in enumerate(metrics):
        ax = axes[i]
        sns.barplot(data=df, x="Scenario", y=metric, hue="Method", ax=ax)
        ax.set_title(f"{metric} Across Scaling Scenarios")
        ax.set_xlabel("Scenario")
        ax.set_ylabel(metric)
    plt.tight_layout()
    plt.show()


def plot_distribution(distribution_data, scenario_name):
    fig, ax = plt.subplots(figsize=(10, 5))
    methods = ["ABC", "GEN"]
    width = 0.35
    for i, method in enumerate(methods):
        dist = distribution_data[(scenario_name, method)]
        vms = sorted(dist.keys())
        loads = [dist[vm] for vm in vms]
        ax.bar([x + i * width for x in range(len(vms))], loads, width, label=method)
    ax.set_xticks([x + width / 2 for x in range(len(vms))])
    ax.set_xticklabels(vms, rotation=90)
    ax.set_ylabel("Total Load (Task Weight)")
    ax.set_title(f"Task Distribution Across VMs - {scenario_name}")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    scenarios = {
        "Small (50×10)":  (50, 10),
        "Medium (100×20)": (100, 20),
        "Large (250×50)":  (250, 50),
        # "XL (500×100)":    (500, 100),
        # "XXL (1000×200)":  (1000, 200),
    }
    df_results, distribution_data = run_all_scenarios(scenarios)
    print(df_results)
    plot_multiscenario(df_results)
    plot_distribution(distribution_data, "Medium (100×20)")