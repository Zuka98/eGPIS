from reconstruct import run_experiment
from Parameters import Parameters
from munch import Munch
import os
import time


def configure_paths(experiment_name, dataset_name, params):
    #Path configurations
    dataset_location = "../datasets"
    dataset_path = os.path.join(dataset_location, dataset_name)
    gt_path = f'{dataset_path}_gt'

    #Experiment Parameters
    results_path = os.path.join("../results", dataset_name)
    output_path = os.path.join(results_path, f"{experiment_name}_{params.stringify()}")
    log_path = os.path.join(output_path, "logs")
    json_path = os.path.join(output_path, "json")
    summary_path = os.path.join(output_path, "summary")
    objects_path = os.path.join(output_path, "objects")

    #Manage results directories
    if not os.path.isdir(results_path):
        os.mkdir(results_path)

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    if not os.path.isdir(log_path):
        os.mkdir(log_path)

    if not os.path.isdir(json_path):
        os.mkdir(json_path)

    if not os.path.isdir(summary_path):
        os.mkdir(summary_path)

    if not os.path.isdir(objects_path):
        os.mkdir(objects_path)

    return Munch({
        "dataset_path": dataset_path,
        "results_path": results_path,
        "log_path": log_path,
        "json_path": json_path,
        "objects_path": objects_path,
        "gt_path": gt_path,
        "summary_path": summary_path
    })


if __name__ == "__main__":
    experiment_name = "experiment1"
    dataset_name = "famous"
    global_points, local_points = 6000, 8000
    global_iterations,local_iterations = 300,300
    parameters = Parameters(
        global_points, # Global sample size
        global_iterations, 
        50, # Grid cube size nxnxn
        1, # Octree Depth
        local_points, # Local sample size
        local_iterations,
        "n" # Augmentation method --> s=simple, r=radial, n=normal
    )
    paths = configure_paths(experiment_name, dataset_name, parameters)
    run_experiment(parameters, paths)
    
