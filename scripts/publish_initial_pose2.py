#!/usr/bin/env python3
# coding=utf8
from __future__ import print_function, division, absolute_import

import argparse

import rclpy
from rclpy.node import Node
import tf_transformations
from geometry_msgs.msg import Pose, Point, Quaternion, PoseWithCovarianceStamped


class PublishInitialPose2(Node):
    def __init__(self, x, y, z, yaw, pitch, roll):
        super().__init__('publish_initial_pose')
        self.pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        quat = tf_transformations.quaternion_from_euler(roll, pitch, yaw)
        xyz = [x, y, z]
        initial_pose = PoseWithCovarianceStamped()
        initial_pose.pose.pose = Pose(Point(*xyz), Quaternion(quat[0], quat[1], quat[2], quat[3]))
        initial_pose.header.frame_id = 'map'
        self.timer = self.create_timer(1.0, lambda: self.publish_and_exit(initial_pose))

    def publish_and_exit(self, msg: PoseWithCovarianceStamped):
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(msg)
        self.get_logger().info('Initial Pose published')
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('x', type=float)
    parser.add_argument('y', type=float)
    parser.add_argument('z', type=float)
    parser.add_argument('yaw', type=float)
    parser.add_argument('pitch', type=float)
    parser.add_argument('roll', type=float)
    args = parser.parse_args()

    rclpy.init()
    node = PublishInitialPose2(args.x, args.y, args.z, args.yaw, args.pitch, args.roll)
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
