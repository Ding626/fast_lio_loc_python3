#!/usr/bin/env python3
"""
Simple node that republishes an Odometry message as a TF between 'odom' and 'base_footprint'.
"""
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry


def publish_odom_tf(odom_msg):
    transform = TransformStamped()
    transform.header.stamp = rospy.Time.now()  # 关键：使用当前时间戳[1,6](@ref)
    transform.header.frame_id = "t265_odom_frame"
    transform.child_frame_id = "base_footprint"
    transform.transform.translation.x = odom_msg.pose.pose.position.x
    transform.transform.translation.y = odom_msg.pose.pose.position.y
    transform.transform.translation.z = odom_msg.pose.pose.position.z
    transform.transform.rotation = odom_msg.pose.pose.orientation

    tf_broadcaster.sendTransform(transform)


if __name__ == '__main__':
    rospy.init_node('odom_tf_publisher')
    tf_broadcaster = tf2_ros.TransformBroadcaster()
    rospy.Subscriber('/t265/odom/sample', Odometry, publish_odom_tf)
    rospy.spin()
