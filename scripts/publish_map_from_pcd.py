#!/usr/bin/env python3
"""Publish a PCD file as a ROS2 `/map` PointCloud2 message (transient_local QoS).

Usage:
  ros2 run fast_lio_localization publish_map_from_pcd.py /path/to/map.pcd
  or
  python3 scripts/publish_map_from_pcd.py /path/to/map.pcd

This node uses Open3D to read the PCD and `sensor_msgs_py.point_cloud2.create_cloud`
to build a PointCloud2 message. The publisher uses TRANSIENT_LOCAL durability so
late subscribers can receive the map.
"""

from __future__ import annotations
import os
import argparse
import numpy as np
import open3d as o3d

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header


class MapPublisher(Node):
    def __init__(self, pcd_path: str, topic: str = '/map', publish_rate: float = 1.0):
        super().__init__('map_publisher_from_pcd')

        self.pcd_path = pcd_path
        self.topic = topic

        qos = QoSProfile(depth=1)
        qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        qos.reliability = QoSReliabilityPolicy.RELIABLE

        self.pub = self.create_publisher(PointCloud2, self.topic, qos)
        self.timer = self.create_timer(1.0 / max(1e-3, publish_rate), self.timer_cb)
        self.pc_msg = None
        self.published_once = False

        self.get_logger().info(f'Initialized map publisher: pcd={pcd_path} topic={topic} rate={publish_rate}Hz')

        try:
            self.pc_msg = self.load_pcd_as_pointcloud2(pcd_path)
            if self.pc_msg is None:
                self.get_logger().error('Failed to load PCD file into PointCloud2')
        except Exception as e:
            self.get_logger().error(f'Exception while loading PCD: {e}')
            self.pc_msg = None

    def load_pcd_as_pointcloud2(self, path: str) -> PointCloud2 | None:
        if not os.path.exists(path):
            self.get_logger().error(f'PCD file not found: {path}')
            return None

        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points)
        if pts.size == 0:
            self.get_logger().warning('Loaded PCD has 0 points')

        # Build PointField list and rows. If colors exist, include intensity.
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        rows = []
        if hasattr(pcd, 'colors') and len(np.asarray(pcd.colors)) == pts.shape[0]:
            # derive intensity from RGB luminance if colors are present
            colors = np.asarray(pcd.colors)
            intensity = (0.2989 * colors[:, 0] + 0.5870 * colors[:, 1] + 0.1140 * colors[:, 2]).astype(np.float32)
            fields.append(PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1))
            rows = [ (float(x), float(y), float(z), float(i)) for (x, y, z), i in zip(pts, intensity) ]
        else:
            rows = [ (float(x), float(y), float(z)) for (x, y, z) in pts ]

        header = Header()
        header.frame_id = 'map'
        header.stamp = self.get_clock().now().to_msg()

        msg = point_cloud2.create_cloud(header, fields, rows)
        # Ensure message metadata explicitly encodes number of points so downstream
        # consumers can read it without parsing payload bytes. create_cloud often
        # sets width/height, but be explicit here to avoid ambiguity for callers.
        try:
            n_points = int(len(rows))
            msg.width = n_points
            msg.height = 1
            # point_step: size of each point in bytes. All fields are FLOAT32 -> 4 bytes
            point_step = 4 * len(fields)
            msg.point_step = point_step
            msg.row_step = point_step * msg.width
            msg.is_dense = True
            # is_bigendian left as whatever create_cloud provided (usually False)
        except Exception:
            # If anything unexpected occurs, fall back to the message produced
            # by create_cloud and continue — don't raise here.
            pass
        # Keep a copy with current header; timer_cb will refresh stamp
        return msg

    def timer_cb(self):
        if self.pc_msg is None:
            return
        # update timestamp each publish so subscribers see a fresh stamp
        self.pc_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self.pc_msg)
        if not self.published_once:
            # Log descriptive info on first publish
            try:
                # try to compute number of points from data length and fields
                npoints = None
                # if height==1 and width set, use width
                npoints = getattr(self.pc_msg, 'width', None)
                if npoints is None or npoints == 0:
                    # best effort: derive from payload size / point size
                    point_step = getattr(self.pc_msg, 'point_step', 0)
                    if point_step > 0:
                        npoints = int(len(self.pc_msg.data) / point_step)
                self.get_logger().info(f'Published /map, approx points={npoints}')
            except Exception:
                self.get_logger().info('Published /map')
            self.published_once = True


def main():
    parser = argparse.ArgumentParser(description='Publish a PCD as /map PointCloud2 (transient_local)')
    parser.add_argument('pcd', help='path to .pcd file')
    parser.add_argument('--topic', default='/map', help='topic to publish (default: /map)')
    parser.add_argument('--rate', type=float, default=1.0, help='publish rate in Hz (default: 1.0)')
    # Use parse_known_args so ROS2 `--ros-args` and remapping args injected by
    # the launch system don't cause argparse to fail. Unknown args are ignored.
    args, unknown = parser.parse_known_args()

    if not args.pcd:
        parser.print_usage()
        raise SystemExit('PCD file path is required')

    rclpy.init()
    node = MapPublisher(args.pcd, topic=args.topic, publish_rate=args.rate)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
