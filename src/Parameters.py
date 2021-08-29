class Parameters():
    def __init__(self, global_density, global_iters, grid_size, octree_depth=0, local_density=0, local_iters=0, augmentation_method="n"):
        self.global_density = global_density
        self.global_iters = global_iters
        self.grid_size = grid_size
        self.octree_depth = octree_depth
        self.local_density = local_density
        self.local_iters = local_iters
        self.augmentation_method = augmentation_method


    def get_parameters(self):
        return self.global_density,self.global_iters,self.grid_size,self.octree_depth,self.local_density,self.local_iters, self.augmentation_method

    def get_info(self):
        return (
            f"\nDensity: {self.global_density}\n" \
            f"Iterations: {self.global_iters}\n" \
            f"Grid Size: {self.grid_size}\n" \
            f"Octree Depth: {self.octree_depth}\n" \
            f"Local Density: {self.local_density}\n" \
            f"Local Iterations: {self.local_iters}\n" \
            f"Augmentation: {self.augmentation_method}"
        )

    def stringify(self):
        name = f"({self.global_density},{self.global_iters},{self.grid_size},{self.octree_depth}"
        if self.octree_depth:
            name += f",{self.local_density},{self.local_iters}"
            if self.augmentation_method != "n":
                name += ",n"
        name += ')'
        return name