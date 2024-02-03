import copy
import os
import torch

def flatten_params(model):
  return model.state_dict()

def lerp(lam, t1, t2):
  t3 = copy.deepcopy(t2)
  for p in t1: 
    t3[p] = (1 - lam) * t1[p] + lam * t2[p]
  return t3

"""
Saves model checkpoint in "checkpoint_dir/experiment_name/checkpoint_name"
"""
def save_model(model, checkpoint_name, experiment_name, checkpoint_dir="checkpoints"):
    checkpoint_save_dir = os.path.join(checkpoint_dir, experiment_name)
    os.makedirs(checkpoint_save_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_save_dir, checkpoint_name)

    print(f"Saving Checkpoint: {checkpoint_path}")

    torch.save(model.state_dict(), checkpoint_path)