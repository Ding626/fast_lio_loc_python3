#!/usr/bin/env python3
"""
swap_coordinates.py

Rotate a PCD's points. By default it performs a 90° rotation around Y then
a 90° rotation around X (applied after the Y rotation).

Usage:
    python3 swap_coordinates.py in.pcd [out.pcd] [deg_y] [deg_x]

If output path is omitted, writes to ./map_swapped.pcd
"""
import sys
import os

try:
    import open3d as o3d
    import numpy as np
except Exception as e:
    print('Missing dependency:', e)
    print('Install with: pip3 install open3d numpy')
    sys.exit(1)


def rotate_points(points: np.ndarray, deg_y: float = 90.0, deg_x: float = 90.0, deg_z: float = 180.0) -> np.ndarray:
    """Rotate the point cloud: Y by `deg_y`, then X by `deg_x`, then Z by `deg_z`.
    Returns an Nx3 (or NxM if input had extra columns) array with rotated coordinates.
    """
    if points.shape[1] < 3:
        raise ValueError('Point array must have at least 3 columns for x,y,z')

    # Rotation around Y
    ty = np.deg2rad(deg_y)
    cy = np.cos(ty)
    sy = np.sin(ty)
    R_y = np.array([[cy, 0.0, sy],
                    [0.0, 1.0, 0.0],
                    [-sy, 0.0, cy]])

    # Rotation around X
    tx = np.deg2rad(deg_x)
    cx = np.cos(tx)
    sx = np.sin(tx)
    R_x = np.array([[1.0, 0.0, 0.0],
                    [0.0, cx, -sx],
                    [0.0, sx, cx]])

    # Rotation around Z (applied last)
    tz = np.deg2rad(deg_z)
    cz = np.cos(tz)
    sz = np.sin(tz)
    R_z = np.array([[cz, -sz, 0.0],
                    [sz, cz, 0.0],
                    [0.0, 0.0, 1.0]])

    # apply Y then X then Z: p' = R_z * (R_x * (R_y * p)) => R = R_z @ R_x @ R_y
    R = R_z.dot(R_x.dot(R_y))
    pts_rot = points[:, :3].dot(R.T)

    # preserve extra columns if present
    if points.shape[1] > 3:
        rest = points[:, 3:]
        return np.hstack([pts_rot, rest])
    return pts_rot


def main():
    if len(sys.argv) < 2:
        print('Usage: swap_coordinates.py in.pcd [out.pcd]')
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.getcwd(), 'map_swapped.pcd')

    if not os.path.exists(in_path):
        print('Input file does not exist:', in_path)
        sys.exit(1)

    pcd = o3d.io.read_point_cloud(in_path)
    if pcd.is_empty():
        print('Input point cloud is empty')
        sys.exit(1)

    pts = np.asarray(pcd.points)
    # optional arguments: degrees around Y, X and Z (defaults 90,90,180)
    deg_y = float(sys.argv[3]) if len(sys.argv) > 3 else 90.0
    deg_x = float(sys.argv[4]) if len(sys.argv) > 4 else 90.0
    deg_z = float(sys.argv[5]) if len(sys.argv) > 5 else 180.0
    pts_swapped = rotate_points(pts, deg_y=deg_y, deg_x=deg_x, deg_z=deg_z)

    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(pts_swapped)

    ok = o3d.io.write_point_cloud(out_path, pcd_out)
    if ok:
        print('Wrote swapped point cloud to', out_path)
    else:
        print('Failed to write point cloud to', out_path)


if __name__ == '__main__':
    main()
