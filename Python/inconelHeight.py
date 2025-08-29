#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 09:07:55 2025

@author: Rodrigo Bostrom
"""

import open3d as o3d
import numpy as np
import tkinter as tk
from tkinter import filedialog
from sklearn.cluster import KMeans

def get_pcd_filepath():
    print("Opening file selection dialog...")
    root = tk.Tk()
    root.withdraw()  
    filepath = filedialog.askopenfilename(
        title="Select the PCD file with the deposited bead",
        filetypes=(("Point Cloud Data", "*.pcd"), ("All files", "*.*"))
    )
    root.destroy()
    if filepath:
        print(f"File selected: {filepath}")
    else:
        print("File selection cancelled.")
    return filepath

# Measuring the plane height
def get_plane_height(plane_model, inlier_points):
    a, b, c, d = plane_model
    if np.isclose(c, 0): return np.mean(inlier_points[:, 2])
    centroid = np.mean(inlier_points, axis=0)
    # Isolating z in a*x + b*y + c*z + d = 0
    z_height = (-a * centroid[0] - b * centroid[1] - d) / c
    return z_height

# -- Importing file --
pcd_file = get_pcd_filepath()
pcd    = o3d.io.read_point_cloud(pcd_file)
points = np.asarray(pcd.points)

# -- Filtering image to remove outliers -- 
z_min = 0.08                            # Minimum Z to keep (adjust to your data)
center_x, center_y = -0.003, 0.003      # Set center coordinates
inner_radius = 0.018                    # Screw radius

# Z Filter
mask_z = points[:, 2] >= z_min

# Screw area removal
distances = np.sqrt((points[:, 0] - center_x)**2 + (points[:, 1] - center_y)**2)
mask_inner_circle = distances >= inner_radius

# Combine filters
mask = mask_z & mask_inner_circle
filtered_points = points[mask]

# Defining pointcloud based on defined masks
filtered_pcd = o3d.geometry.PointCloud()
filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

# Visualization
# o3d.io.write_point_cloud("substract_filtered.pcd", filtered_pcd)
# o3d.visualization.draw_geometries([filtered_pcd])

# -- Removing substrate --
distance_threshold = 0.0025
ransac_n = 3                 
num_iterations = 5000        

# Perform plane segmentation to find the plane model and the indices of inlier points
plane_model, inlier_indices = filtered_pcd.segment_plane(distance_threshold=distance_threshold,
                                               ransac_n=ransac_n,
                                               num_iterations=num_iterations)

[a, b, c, d] = plane_model

# This pcd will have all points that are NOT in plane model 
cylinder_cloud = filtered_pcd.select_by_index(inlier_indices, invert=True)

# Substrate pointcloud; pointclouds in substrate
substrate_cloud = filtered_pcd.select_by_index(inlier_indices)
substrate_cloud.paint_uniform_color([0, 0, 1.0])

o3d.visualization.draw_geometries(
    [cylinder_cloud], 
    window_name="Verification - Press 'q' to close"
)

print(f"Substrate height with respect to KP2 Tool0: {np.abs(d):.4f} m")

# -- Measuring the inconel height --
z_substrate = np.abs(d)

all_points = np.asarray(cylinder_cloud.points)
z_coords = all_points[:, 2].reshape(-1, 1)

# Applying K-Means to cluster heights; lowest (inconel) and highest (steel)
# n_clusters: The number of clusters to form as well as the number of centroids to generate.
# random_state: Determines random number generation for centroid initialization
# n_init: Number of times the k-means algorithm is run with different centroid seeds.
kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto')
labels = kmeans.fit_predict(z_coords)

mean_z_0 = np.mean(all_points[labels == 0, 2])
mean_z_1 = np.mean(all_points[labels == 1, 2])

# Inconel label has the lowest medium height
inconel_label = 0 if mean_z_0 < mean_z_1 else 1

inconel_indices = np.where(labels == inconel_label)[0]
inconel_cloud = cylinder_cloud.select_by_index(inconel_indices)

# Coloring both groups to visualize the clusters
steel_indices = np.where(labels != inconel_label)[0]
steel_cloud = cylinder_cloud.select_by_index(steel_indices)
inconel_cloud.paint_uniform_color([1, 0, 0])     # Red for inconel
steel_cloud.paint_uniform_color([0.5, 0.5, 0.5]) # Gray for steel
o3d.visualization.draw_geometries([inconel_cloud], window_name="Clusters: Inconel (Red) e Steel (Gray)")

inconel_plane_model, inconel_inlier_indices = inconel_cloud.segment_plane(
    distance_threshold=0.001,
    ransac_n=3,
    num_iterations=1000
)

inconel_cloud = inconel_cloud.select_by_index(inconel_inlier_indices)
inconel_cloud.paint_uniform_color([0, 1, 0])
o3d.visualization.draw_geometries([inconel_cloud], window_name="Inconel")

# Calculating inconel height
z_inconel = get_plane_height(inconel_plane_model, np.asarray(inconel_cloud.points))
print(f"--> Inconel height (Z_inconel): {z_inconel:.4f} m")
print(f"--> Substrate height (Z_substrate): {z_substrate:.4f} m")

# Calcula a altura final do cordão em relação ao substrato
bead_height = z_inconel - z_substrate

print("\n" + "="*50)
print("Analysis completed!")
print(f"  -> Height of inconel bead: {bead_height * 1000:.3f} mm")
print("="*50)
