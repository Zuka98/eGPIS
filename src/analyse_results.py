if __name__ == "__main__":
    experiment_name = "experiment1"
    dataset_name = "famous_ply"
    
    dataset_location = "../datasets"
    text_filename = "files.txt"
    dataset_path = os.path.join(dataset_location, dataset_name)
    gt_path = os.path.join(dataset_location, 'gt')
    list_file = open(os.path.join(dataset_path,text_filename), "r")

    results_path = os.path.join('../results', dataset_name, experiment_name)
    
    #Experiment Parameters
    params = Parameters(1000,100,60,0,1500,150,"n")

    results_path = os.path.join("../results", dataset_name)
    output_path = os.path.join(results_path, f"{experiment_name}_{params.stringify()}")
    log_path = os.path.join(output_path, "logs")
    logfile = None

    #Manage results directories
    if not os.path.isdir(results_path):
        os.mkdir(results_path)
    
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    
    if not os.path.isdir(log_path):
        os.mkdir(log_path)

    results = []
    for name in list_file:
        #Load File
        filename = name.strip()
        pcd_path = os.path.join(dataset_path, filename + ".ply")
        pcd = o3d.io.read_point_cloud(pcd_path)
        
        print("Training:",filename)
        
        logfile = open(f"{log_path}/{name.strip()}.log","w")
        #Octree Depth 0 means no subdivisioning. 
        logfile.write(
            f"Model Info:\n" \
            f"Object Name: {name}\n" \
            f"{params.get_info()}\n"
         )
        #Reconstruct
        
        result = reconstruct(pcd, params)
        result.name = filename
        #Save mesh into results folder
        if result.pred_mesh:
            o3d.io.write_triangle_mesh(f'{str(output_path)}/{filename}.obj', result.pred_mesh)
        else:
            logfile.write("\nFailed to reconstruct object\n")
            raise Exception("Failed to reconstruct object")
        
        #Try to load GT mesh
        try:
            gt_mesh_path = os.path.join(gt_path, filename + '.obj')
            result.gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_path)
        except Exception:
            result.gt_mesh = None
            print("GT object for", filename, "Not Found")

        #Compare Predicted Mesh vs GT and return Chamfer D, SA and G.T S.A
        logfile.write(str(result.get_summary()))
        logfile.close()

        results.append(result.get_summary())
    
    dataframe = pd.DataFrame(results, columns = ["Name", "Chamfer D.", "GT S.A.", "Pred. S.A.", "T. Points",  "T. Time(m)", "R. Time(m)"])
    #Save Results
    with open(f'{log_path}/results.tex', 'w') as file:
        file.write(dataframe.to_latex())
    
    with open(f'{log_path}/results.csv', 'w') as file:
        file.write(dataframe.to_csv())


    list_file.close()