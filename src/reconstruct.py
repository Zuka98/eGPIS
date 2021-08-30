import sys
import numpy as np
import open3d as o3d
from sklearn.preprocessing import StandardScaler
import torch
import gpytorch
import time
from skimage import measure
import os
import pandas as pd
from Parameters import Parameters
from Reconstruction import Reconstruction
from GPRegressionModel import GPRegressionModel
from helpers import *
import json
# from chamferdist import ChamferDistance

'''
1. Provide the path to the txt file containing object list in dataset folder: "files.txt"
   Dataset must comprise of .ply files (raw pointcloud observations)

2. Load objects using Open3D library (can be done by any other library but this provides 
   the most convinient way + includes other useful functions)

3. Call the reconstruction function with parameters 
   (pointcloud (o3d) data, desired density, iterations, logfile pointer, grid size, augmentation method) 

4. Save the outputs
'''
def train(name, train_x, train_y, logfile, iterations = 200):
    print(f"\n\n\nTraining {name} Model. Training Points = {len(train_x)}\n")
    logfile.write(f"\n\n\nTraining {name} Model. Training Points = {len(train_x)}\n")
    start = time.time()
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPRegressionModel(train_x, train_y, likelihood)
    model.train(), likelihood.train()
    # # # Choose Either ADAM or SGD for best results
    # optimizer = torch.optim.LBFGS(model.parameters())
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01) 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Includes GaussianLikelihood parameters
    # # # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for i in range(0, iterations + 1):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        logfile.write('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f \n' % (
            i, iterations, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
        print('%s: Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            name, i, iterations, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
        
        optimizer.step()
        
        # if (abs(loss) <= 0.1):
        #     print("Regression converged")
        #     break
            
    end = time.time()
    total_time = (end - start) / 60
    logfile.write(f"\nTraining Time: {total_time}\n\n")
    print('\n')

    return model, likelihood


def evaluate(model, likelihood, test_x):
    model.eval()
    likelihood.eval()
    #Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))
    
    mean = observed_pred.mean.numpy()
    variance = observed_pred.variance.detach().numpy()
    return mean,variance


def get_target_density(n_points, density):
    return 2 * n_points // density - 1


def get_variances(verts, model, likelihood, sc):
    transformed = torch.tensor(verts).float()
    _, variances = evaluate(model,likelihood, transformed)
    return variances

'''
pcd -> Instance of open3d library pointcloud
density -> Desired number of training points (including augmented poitns) 1500 <= density <= 11000 (Depending on the RAM, as it runs Gaussian Process too many points may cause it to crash)
iterations -> maximum number of iterations to be taken by GP
grid_size -> determines the size of cubic grid that is evaluated for object reconstruction. Higher the number, better the reconstruction fidelity. (but it increases computation demand)
augmentation_method -> s=simple, r=radial, n=normal. If normals are not available specify simple/radial as an augmentation method, generally radial should work better 
octree_depth -> Number of local regions in sub-divided space. 
local_density -> Dictates what should be the maximum number of points in each region of Octree
'''
def reconstruct(pcd, experiment, filename, logfile):
    reconstruction = Reconstruction()

    density, global_iters, grid_size, octree_depth, local_density, local_iters, augmentation_method = experiment.get_parameters()
    vertices, normals = np.asarray(pcd.points), np.asarray(pcd.normals)
    
    # # # Visualize Loaded Point Cloud
    # scatter_points(vertices, size=2, opacity=1)

    # # # Fit Transform, and Translate Data
    sc = StandardScaler()
    fitted_data = sc.fit_transform(vertices)

    x,y,z = zip(*fitted_data)
    x_min, y_min, z_min = min(x), min(y), min(z)
    x_shift, y_shift, z_shift = 0,0,0
    if (x_min < 0):
        x_shift = abs(x_min)

    if (y_min < 0):
        y_shift = abs(y_min)

    if (z_min < 0):
        z_shift = abs(z_min)

    x = x + x_shift
    y = y + y_shift
    z = z + z_shift
    
    transformed_vertices = np.dstack((x,y,z))[0]


    # # Visualise Transormed Point Cloud
    # scatter_points(transformed_vertices, size=2, opacity=1)
    # # # Augment Data. 
    # # # For GP keep threshold 0.004 when applied on raw data, or 0.025 when applied on augmented data
    points, labels, points_sub, labels_sub = [], [], [], []
    used_normal_augmentation = False
    if len(normals) == len(vertices) and augmentation_method == 'n':
        used_normal_augmentation = True
        #Segment Critical Set First. Ommited takes too long
        # critical_v, critical_n, arbitrary_v, arbitrary_n = segment_critical_set(transformed_vertices, normals, 500)
        #Augment Points
        augmented_points, augmented_labels = normal_augmentation(transformed_vertices, normals, threshold=0.035)
        points = np.concatenate((augmented_points, transformed_vertices), axis=0)
        labels = np.concatenate((augmented_labels, [0] * len(transformed_vertices)),axis=0) 

        target_density = get_target_density(len(points), density)
        points_sub, labels_sub = downsample_uniform(points, labels, target_density)

        #Combine first and downsample later (With critical set segmentation)
        #Take 1/3 of total points to be critical set and 2/3 all augmented ones
        # target_density_arbitrary = get_target_density(len(points), density * 2 // 3)
        # target_density_critical =  get_target_density(len(critical_v), density * 1 // 3)
        # points_sub_arbitrary, labels_sub_arbitrary = downsample_uniform(points,labels,target_density_arbitrary)
        # points_sub_critical, labels_sub_critical = downsample_uniform(critical_v,[0] * len(critical_v),target_density_critical)
        # points_sub = np.concatenate((points_sub_arbitrary, points_sub_critical), axis=0)
        # labels_sub = np.concatenate((labels_sub_arbitrary, labels_sub_critical), axis=0)
    else:
        augmented_points, augmented_labels = radial_augmentation(transformed_vertices, threshold=0.005)
        
        target_density = get_target_density(len(transformed_vertices) + len(augmented_points), density)
        points_sub, labels_sub = downsample_uniform(transformed_vertices, [0 for x in transformed_vertices],target_density)

        points_sub = np.concatenate((points_sub, augmented_points), axis=0)
        labels_sub = np.concatenate((labels_sub, augmented_labels), axis=0)
  
    # # Visualize Augmented Data
    # scatter_points(points_sub, size=1.5, opacity=1, color=labels)

    # # # Generate test grid using scaled data
    grid = generate_grid(transformed_vertices,size=grid_size)

    # # # initialize likelihood and model
    train_x = torch.tensor(points_sub).float()
    train_y = torch.tensor(labels_sub).float()
    logfile.write(f"Total Training Points: {len(train_x)}\n")
    reconstruction.train_points = len(train_x)
    
    # # # Learn Implicit Function using GP
    reconstruction.train_start = time.time()
    model, likelihood = train(f"{filename} Global:", train_x, train_y, logfile, iterations = global_iters)
    reconstruction.train_end = time.time()
    print("Training Global Shape Compelted")
    
    
    test_x = torch.tensor(grid).float()
    predictions, variances = evaluate(model,likelihood, test_x)
    lengthscale = model.covar_module.base_kernel.lengthscale.item()


    # # # Visualize Predicted Pointcloud
    # visualize_output_with_threshold(test_x,y_pred,threshold=0.25,color=variance)
    print(f"Trained: {filename}, Vertices: {len(vertices)}, Sample Size: {len(train_x)}")
    print(f"Min Prediction = {min(predictions)}, Max Prediction = {max(predictions)} Length Scale = {lengthscale}")

    # # # Incorporate Variance into improving reconstruction accuracy. Get's rid of points where model is not confident 
    # filtered_pred = y_pred.copy()
    # for i in range(len(variance)):
    #     if variance[i] > 2:
    #         filtered_pred[i] += variance[i] * 3
    # y_pred = filtered_pred
    
    # # # * * *  Octree Local Region Fitting. Only kicks in if octree_depth > 0 * * * # # # 
    if used_normal_augmentation and augmentation_method == 'n' and octree_depth:
        reconstruction.train_points = 0 #This is because we are re-using local poitns
        # # Create Octree Sub-divisioning
        # Regions to train
        pcd.points = o3d.utility.Vector3dVector(points)
        octree = o3d.geometry.Octree(max_depth=octree_depth)
        octree_size_expand = 0.01
        octree.convert_from_point_cloud(pcd, size_expand=octree_size_expand)
        # # Visualize Octree Partitioning of the Object. (Cubic shape corresponds to level 1)
        # o3d.visualization.draw_geometries([octree])
        root = octree.root_node
        # node_indices = root.indices
        # scatter_points(vertices[node_indices], size=2, opacity=1)

        gps = dict()
        scalers = dict()

        # Regions to Evaluate (Grid Octree)
        pcd_grid = o3d.geometry.PointCloud()
        pcd_grid.points = o3d.utility.Vector3dVector(grid)
        octree_grid = o3d.geometry.Octree(max_depth=octree_depth)
        octree_grid.convert_from_point_cloud(pcd_grid, size_expand=octree_size_expand)
        grid_root = octree_grid.root_node
        local_predictions, local_variances = np.ones(len(grid)), np.ones(len(grid)) * 5
        #Training
        for i in range(0, len(root.children)): 
            print(i, root.children[i])
            if not root.children[i]:
                continue
            
            indices = root.children[i].indices
            v = points[indices]
            l = labels[indices]
            target_density = get_target_density(len(v),local_density)

            down_x, down_y = v, l
            #If sample is dense enough then downsample
            if len(v) > 1500 and target_density > 5:
                down_x, down_y = downsample_uniform(v, l, target_density)

            # scalers[i] = StandardScaler()
            # fitted_v = scalers[i].fit_transform(v)
            ### Sort the points
            local_x, local_y = torch.tensor(down_x).float(), torch.tensor(down_y).float()
            reconstruction.train_points += len(local_x)
            gps[i] = train(f"{filename} Local {i}", local_x, local_y, logfile, iterations = local_iters)
            
        #Evaluate
        for i in range(len(grid_root.children)):
            # test_points = scalers[i].transform(grid[indices])
            if i not in gps:
                continue

            indices = grid_root.children[i].indices
            local_test_x = torch.tensor(grid[indices]).float()
            model_local, likelihood_local = gps[i]
            local_predictions[indices], local_variances[indices] = evaluate(model_local,likelihood_local, local_test_x)
        
        # scatter_points_with_threshold(grid,predictions,variances,threshold=0.25)
        # scatter_points_with_threshold(grid,local_predictions,local_variances,threshold=0.25)

        # Merge Local and Global Approximations
        for i in range(len(predictions)):
            #Condition is uncertainty. Replace global prediction with the local one iff local variance 
            #is better and difference between prediction is no more than 0.3
            if variances[i] > local_variances[i] and abs(predictions[i] - local_predictions[i] < 0.3):
                variances[i] = local_variances[i]
                predictions[i] = local_predictions[i]



    # # # Using the spacing parameter, data gets scaled back to its original size
    spacing = get_scale(grid, grid_size)
    voxel = np.reshape(predictions, (grid_size, grid_size, grid_size))
    marching_level = 0.035
    v,t,n,_ = measure.marching_cubes( 
                volume = voxel,
                level = marching_level,
                gradient_direction = 'descent',
                spacing=spacing,
                step_size = 1,
            )

    # scatter_points(np.asarray(v), size=2, opacity=1)
    result_pt = sc.inverse_transform(v - [x_shift,y_shift,z_shift])
    # colors = get_variances(v, model, likelihood, sc)

    # # Predicted Mesh
    pred_mesh = o3d.geometry.TriangleMesh()
    pred_mesh.vertices = o3d.utility.Vector3dVector(result_pt)
    pred_mesh.triangles = o3d.utility.Vector3iVector(t)
    reconstruction.pred_mesh = pred_mesh
    reconstruction.end = time.time()

    # # # Visualize reconstructed mesh - Colors correspond to GP variance (JET color scheme)
    # show_mesh(result_pt,t, colors)

    return reconstruction


def run_experiment(params, paths):
    text_filename = "files.txt"
    list_file = open(os.path.join(paths.dataset_path, text_filename), "r")
    results = []

    for name in list_file:
        #Load File
        filename = name.strip()
        pcd_path = os.path.join(paths.dataset_path, filename + ".ply")
        pcd = o3d.io.read_point_cloud(pcd_path)
        
        print("Training:", filename)
        
        logfile = open(f"{paths.log_path}/{name.strip()}.log","w")
        #Octree Depth 0 means no subdivisioning. 
        logfile.write(
            f"Model Info:\n" \
            f"Object Name: {name}\n" \
            f"{params.get_info()}\n"
         )

        #Reconstruct
        result = reconstruct(pcd, params, filename, logfile)
        result.name = filename

        #Save mesh into results folder
        if result.pred_mesh:
            o3d.io.write_triangle_mesh(f'{str(paths.objects_path)}/{filename}.obj', result.pred_mesh)
        else:
            logfile.write("\nFailed to reconstruct object\n")
            raise Exception("Failed to reconstruct object")
        
        #Try to load GT mesh
        try:
            gt_mesh_path = os.path.join(paths.gt_path, filename + '.obj')
            result.gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_path)
        except Exception:
            result.gt_mesh = None
            print("GT object for", filename, "Not Found")

        #Compare Predicted Mesh vs GT and return Chamfer D, SA and G.T S.A
        logfile.write(str(result.get_summary()))
        logfile.close()

        with open(f'{paths.json_path}/{filename}.json', 'w') as file:
            json.dump(result.get_json_summary(), file)

        results.append(result.get_summary())
    
    dataframe = pd.DataFrame(results, columns = ["Name", "Chamfer D.", "GT S.A.", "Pred. S.A.", "T. Points",  "T. Time(m)", "R. Time(m)"])
    
    #Save Results
    with open(f'{paths.summary_path}/results.tex', 'w') as file:
        file.write(dataframe.to_latex())
    
    with open(f'{paths.summary_path}/results.csv', 'w') as file:
        file.write(dataframe.to_csv())

    with open(f'{paths.summary_path}/results.json', 'w') as file:
        json.dump(dataframe.to_json(), file)

    
    list_file.close()