import math
import numpy as np
import sys
import typer

app = typer.Typer()

@app.command()
def scheduler(num_epochs:int,
            pruning_start_epoch:int,
            prune_every_epoch:int,
            sparsity_target:float, 
            prune_schedule:str):
    if pruning_start_epoch >= num_epochs:
        pruning_start_epoch = num_epochs - 1
    num_pruning_steps = math.ceil((num_epochs - pruning_start_epoch) / prune_every_epoch)  # how many times will we have a non-zero prune ptg
    
    schedule = np.zeros(num_epochs)

    if prune_schedule == "linear":
        # Linear schedule: increase sparsity by a fixed percentage each epoch
        ptg = sparsity_target / num_pruning_steps  # how much to increase sparsity on each non-zero pruning step
        for i in range(num_epochs):
            schedule[i] = ptg * (i >= pruning_start_epoch and (i - pruning_start_epoch) % prune_every_epoch == 0)
    elif prune_schedule == "agp":
        cumulative_pruning = [(sparsity_target * (1 - (1 - i / num_pruning_steps) ** 3))  for i in range(1, num_pruning_steps + 1)]
        increments = np.diff([0] + cumulative_pruning)
        schedule[pruning_start_epoch::prune_every_epoch] = increments
    else:
        raise ValueError(f"Unsupported pruning schedule: {prune_schedule}")
    print(f"Pruning schedule: {schedule}\nTotal sparsity: {sum(schedule)}")
    assert(sum(schedule)==sparsity_target)
    return schedule
