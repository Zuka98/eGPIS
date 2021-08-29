import time
from sklearn.preprocessing import StandardScaler
import open3d as o3d

class Reconstruction():
    def __init__(self):
        self.name = "N/A"
        self.start = time.time()
        self.end = time.time()

        self.train_start = time.time()
        self.train_end = time.time()
        self.train_points = 0

        self.pred_mesh = None
        self.gt_mesh = None
        self.chamfer_dist= None

    def get_json_summary(self):
        train_time = "{:.3f}".format((self.train_end - self.train_start) / 60)
        total_time = "{:.3f}".format((self.end - self.start) / 60)
        if not self.chamfer_dist:
            self.get_chamfer_distance(), 
        return {
            "name": self.name, 
            "chamfer_distance": self.chamfer_dist, 
            "gt_surface_area": self.get_surface_area(self.gt_mesh),
            "pred_surface_area": self.get_surface_area(self.pred_mesh),
            "training_points": self.train_points,
            "training_time": train_time,
            "total_time" : total_time
        }



    def get_summary(self):
        train_time = "{:.3f}".format((self.train_end - self.train_start) / 60)
        total_time = "{:.3f}".format((self.end - self.start) / 60)
        if not self.chamfer_dist:
            self.get_chamfer_distance(), 

        return [
            self.name, 
            self.chamfer_dist, 
            self.get_surface_area(self.gt_mesh),
            self.get_surface_area(self.pred_mesh),
            self.train_points,
            train_time,
            total_time
        ]

    def get_chamfer_distance(self):
        if not self.gt_mesh:
            return "N/A"
 
        #Scale the data before comparison
        sc_temp = StandardScaler()
        target = sc_temp.fit_transform(self.gt_mesh.vertices)
        pred = sc_temp.transform(self.pred_mesh.vertices)
        pcl_gt = o3d.geometry.PointCloud()
        pcl_gt.points = o3d.utility.Vector3dVector(target)
        pcl_pred = o3d.geometry.PointCloud()
        pcl_pred.points = o3d.utility.Vector3dVector(pred)
        distance = pcl_gt.compute_point_cloud_distance(pcl_pred)

        self.chamfer_dist = "{:.4f}".format(sum(distance)/len(distance))
        # chamferDist = ChamferDistance()
        # d = chamferDist(torch.tensor([target]).float(), torch.tensor([pred]).float())
        # print(d.detach().item()/len(pred), sum(distance)/len(distance))


    def get_surface_area(self, mesh):
        if mesh:
            return "{:.4f}".format(mesh.get_surface_area())

        return "N/A"