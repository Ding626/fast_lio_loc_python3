#!/usr/bin/env python3
# coding=utf8
from __future__ import print_function, division, absolute_import

import argparse

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
# import  tf2.transformations
import tf2_ros
from geometry_msgs.msg import Pose, Point, Quaternion, PoseWithCovarianceStamped
import numpy as np


def quaternion_from_euler(roll, pitch, yaw):
    # Convert Euler angles (roll, pitch, yaw) to quaternion (x, y, z, w)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([x, y, z, w], dtype=float)


class PublishInitialPose(Node):
    def __init__(self, x, y, z, yaw, pitch, roll):
        super().__init__('publish_initial_pose')
        qos = QoSProfile(depth=1)
        qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        qos.reliability = QoSReliabilityPolicy.RELIABLE
        # publish initial pose transiently so late subscribers (the localization
        # node waiting with TRANSIENT_LOCAL) will receive it
        self.pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', qos)
        quat = quaternion_from_euler(roll, pitch, yaw)
        xyz = [x, y, z]
        initial_pose = PoseWithCovarianceStamped()
        # construct Pose using keyword args to match ROS2 message constructors
        initial_pose.pose.pose = Pose(
            position=Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2])),
            orientation=Quaternion(x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3]))
        )
        initial_pose.header.frame_id = 'map'
        # publish once after short delay to ensure connections
        self.timer = self.create_timer(1.0, lambda: self.publish_and_exit(initial_pose))

    def publish_and_exit(self, msg: PoseWithCovarianceStamped):
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(msg)
        self.get_logger().info('Published initial pose')
        # shutdown after a brief pause
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('x', type=float)
    parser.add_argument('y', type=float)
    parser.add_argument('z', type=float)
    parser.add_argument('yaw', type=float)
    parser.add_argument('pitch', type=float)
    parser.add_argument('roll', type=float)
    # Use parse_known_args so ROS2 `--ros-args` injected by the launch system
    # won't cause argparse to fail when the script is run via `ros2 run`.
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = PublishInitialPose(args.x, args.y, args.z, args.yaw, args.pitch, args.roll)
    try:
        rclpy.spin(node)
    except Exception:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass


if __name__ == '__main__':
    main()
