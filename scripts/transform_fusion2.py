#!/usr/bin/env python3
# coding=utf8
from __future__ import print_function, division, absolute_import

import copy
import time
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import tf_transformations
from geometry_msgs.msg import Pose, Point, Quaternion
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped


cur_odom_to_baselink = None
cur_map_to_odom = None


def pose_to_mat(pose_msg):
    pos = pose_msg.pose.pose.position
    ori = pose_msg.pose.pose.orientation
    T_trans = tf_transformations.translation_matrix((pos.x, pos.y, pos.z))
    T_rot = tf_transformations.quaternion_matrix((ori.x, ori.y, ori.z, ori.w))
    return np.matmul(T_trans, T_rot)


class TransformFusionNode2(Node):
    def __init__(self):
        super().__init__('transform_fusion')
        self.get_logger().info('Transform Fusion Node Inited...')

        self.FREQ_PUB_LOCALIZATION = 50.0
        self.tf_broadcaster = TransformBroadcaster(self)

        qos = QoSProfile(depth=10)
        self.create_subscription(Odometry, '/odom', self.cb_save_cur_odom, qos)
        self.create_subscription(Odometry, '/map_to_odom', self.cb_save_map_to_odom, qos)

        self.pub_localization = self.create_publisher(Odometry, '/localization', qos)

        self.create_timer(1.0 / self.FREQ_PUB_LOCALIZATION, self.timer_cb)

    def cb_save_cur_odom(self, odom_msg):
        global cur_odom_to_baselink
        cur_odom_to_baselink = odom_msg

    def cb_save_map_to_odom(self, odom_msg):
        global cur_map_to_odom
        cur_map_to_odom = odom_msg

    def timer_cb(self):
        global cur_odom_to_baselink, cur_map_to_odom
        cur_odom = copy.copy(cur_odom_to_baselink)
        if cur_map_to_odom is not None:
            T_map_to_odom = pose_to_mat(cur_map_to_odom)
        else:
            T_map_to_odom = np.eye(4)

        # broadcast transform map -> camera_init
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'camera_init'
        trans = tf_transformations.translation_from_matrix(T_map_to_odom)
        quat = tf_transformations.quaternion_from_matrix(T_map_to_odom)
        t.transform.translation.x = float(trans[0])
        t.transform.translation.y = float(trans[1])
        t.transform.translation.z = float(trans[2])
        t.transform.rotation.x = float(quat[0])
        t.transform.rotation.y = float(quat[1])
        t.transform.rotation.z = float(quat[2])
        t.transform.rotation.w = float(quat[3])
        self.tf_broadcaster.sendTransform(t)

        if cur_odom is not None:
            localization = Odometry()
            T_odom_to_base_link = pose_to_mat(cur_odom)
            T_map_to_base_link = np.matmul(T_map_to_odom, T_odom_to_base_link)
            xyz = tf_transformations.translation_from_matrix(T_map_to_base_link)
            quat = tf_transformations.quaternion_from_matrix(T_map_to_base_link)
            localization.pose.pose = Pose(Point(*xyz), Quaternion(*quat))
            localization.twist = cur_odom.twist
            localization.header.stamp = cur_odom.header.stamp
            localization.header.frame_id = 'map'
            localization.child_frame_id = 'body'
            self.pub_localization.publish(localization)


def main(args=None):
    rclpy.init(args=args)
    node = TransformFusionNode2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
