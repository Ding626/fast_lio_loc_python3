#!/usr/bin/env python3
# coding=utf8
from __future__ import print_function, division, absolute_import

import copy
import time
import threading
import tf2_ros
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
# replacement for tf_transformations (use small local helpers)
from geometry_msgs.msg import Pose, Point, Quaternion
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped


cur_odom_to_baselink = None
cur_map_to_odom = None
time_header = None


def pose_to_mat(odom_msg):
    # odom_msg is expected to be nav_msgs/Odometry
    pos = odom_msg.pose.pose.position
    ori = odom_msg.pose.pose.orientation
    T = np.eye(4, dtype=float)
    T[0:3, 3] = np.array([pos.x, pos.y, pos.z], dtype=float)
    # quaternion to rotation matrix
    q = np.array([ori.x, ori.y, ori.z, ori.w], dtype=float)
    # normalize
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    # rotation matrix from quaternion
    R = np.array([
        [1 - 2 * (y * y + z * z),     2 * (x * y - z * w),     2 * (x * z + y * w)],
        [    2 * (x * y + z * w), 1 - 2 * (x * x + z * z),     2 * (y * z - x * w)],
        [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]
    ], dtype=float)
    T[0:3, 0:3] = R
    return T


def translation_from_matrix(T):
    return T[0:3, 3].copy()


def quaternion_from_matrix(T):
    # Convert rotation matrix to quaternion (x, y, z, w)
    R = T[0:3, 0:3]
    # Source: http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S
    return np.array([x, y, z, w], dtype=float)


class TransformFusionNode(Node):
    def __init__(self):
        super().__init__('transform_fusion')
        self.get_logger().info('Transform Fusion Node Inited...')

        self.count = 0
        self.FREQ_PUB_LOCALIZATION = 50.0

        qos = QoSProfile(depth=10)
        self.create_subscription(Odometry, '/odom', self.cb_save_cur_odom, qos)
        self.create_subscription(Odometry, '/map_to_odom', self.cb_save_map_to_odom, qos)

        self.pub_localization = self.create_publisher(Odometry, '/localization', qos)
        self.pub_localization_time = self.create_publisher(Odometry, '/localization_time', QoSProfile(depth=50))

        self.tf_broadcaster = TransformBroadcaster(self)

        # timer for publishing
        period = 1.0 / self.FREQ_PUB_LOCALIZATION
        self.create_timer(period, self.timer_cb)

    def cb_save_cur_odom(self, odom_msg: Odometry):
        global cur_odom_to_baselink
        cur_odom_to_baselink = odom_msg

    def cb_save_map_to_odom(self, odom_msg: Odometry):
        global cur_map_to_odom, time_header
        cur_map_to_odom = odom_msg
        time_header = odom_msg.header.stamp

    def timer_cb(self):
        global cur_odom_to_baselink, cur_map_to_odom
        # copy for thread-safety
        cur_odom = copy.copy(cur_odom_to_baselink)

        if cur_map_to_odom is not None:
            T_map_to_odom = pose_to_mat(cur_map_to_odom)
        else:
            T_map_to_odom = np.eye(4)

        # publish camera_init <- map transform
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'camera_init'
        trans = translation_from_matrix(T_map_to_odom)
        quat = quaternion_from_matrix(T_map_to_odom)
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
            localization_time = Odometry()
            T_odom_to_base_link = pose_to_mat(cur_odom)
            T_map_to_base_link = np.matmul(T_map_to_odom, T_odom_to_base_link)
            xyz = translation_from_matrix(T_map_to_base_link)
            quat = quaternion_from_matrix(T_map_to_base_link)

            localization.pose.pose = Pose(
                position=Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2])),
                orientation=Quaternion(x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3]))
            )
            # copy twist
            localization.twist = cur_odom.twist

            localization.header.stamp = cur_odom.header.stamp
            localization.header.frame_id = 'map'
            localization.child_frame_id = 'base_link'
            self.pub_localization.publish(localization)

            localization_time.pose.pose = Pose(
                position=Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2])),
                orientation=Quaternion(x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3]))
            )
            localization_time.twist = cur_odom.twist
            localization_time.header.stamp = cur_odom.header.stamp
            localization_time.header.frame_id = 'map_1'
            localization_time.child_frame_id = 'base_link'
            self.count += 1
            if self.count >= 5:
                self.pub_localization_time.publish(localization_time)
                self.count = 0


def main(args=None):
    rclpy.init(args=args)
    node = TransformFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
