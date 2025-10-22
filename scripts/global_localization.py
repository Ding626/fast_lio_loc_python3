#!/usr/bin/env python3
# coding=utf8
from __future__ import print_function, division, absolute_import

import copy
import _thread as thread
import time

import open3d as o3d
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import ros_numpy
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Point, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
import numpy as np
import tf_transformations

global_map = None
initialized = False
T_map_to_odom = np.eye(4)
T_map_to_odom_last = np.eye(4)
cur_odom = None
cur_scan = None

# node will be set in main()
node = None


def pose_to_mat(pose_msg):
    """Convert a PoseWithCovarianceStamped or Odometry (or raw Pose) into a 4x4 matrix."""
    if pose_msg is None:
        if node:
            node.get_logger().warning('pose_to_mat received None pose_msg')
        return np.eye(4)

    # pose_msg can be Odometry, PoseWithCovarianceStamped, or Pose
    try:
        pos = pose_msg.pose.pose.position
        ori = pose_msg.pose.pose.orientation
    except Exception:
        try:
            pos = pose_msg.position
            ori = pose_msg.orientation
        except Exception as e:
            if node:
                node.get_logger().error('pose_to_mat: unexpected pose_msg type: %s' % e)
            return np.eye(4)

    T_trans = tf_transformations.translation_matrix((pos.x, pos.y, pos.z))
    T_rot = tf_transformations.quaternion_matrix((ori.x, ori.y, ori.z, ori.w))
    return np.matmul(T_trans, T_rot)
  

def msg_to_array(pc_msg):
    pc_array = ros_numpy.numpify(pc_msg)
    pc = np.zeros([len(pc_array), 3])
    pc[:, 0] = pc_array['x']
    pc[:, 1] = pc_array['y']
    pc[:, 2] = pc_array['z']
    return pc


def registration_at_scale(pc_scan, pc_map, initial, scale):
    scan_ds = voxel_down_sample(pc_scan, SCAN_VOXEL_SIZE * scale)
    map_ds = voxel_down_sample(pc_map, MAP_VOXEL_SIZE * scale)

    # Open3D API changed between versions: registration moved under pipelines.registration
    try:
        reg_module = o3d.registration
        Estimation = reg_module.TransformationEstimationPointToPoint
        Criteria = reg_module.ICPConvergenceCriteria
        registration_icp = reg_module.registration_icp
    except Exception:
        reg_module = o3d.pipelines.registration
        Estimation = reg_module.TransformationEstimationPointToPoint
        Criteria = reg_module.ICPConvergenceCriteria
        registration_icp = reg_module.registration_icp

    result_icp = registration_icp(
        scan_ds, map_ds,
        1.0 * scale, initial,
        Estimation(),
        Criteria(max_iteration=20)
    )

    return result_icp.transformation, result_icp.fitness


def inverse_se3(trans):
    trans_inverse = np.eye(4)
    # R
    trans_inverse[:3, :3] = trans[:3, :3].T
    # t
    trans_inverse[:3, 3] = -np.matmul(trans[:3, :3].T, trans[:3, 3])
    return trans_inverse


def publish_point_cloud(publisher, header, pc):
    data = np.zeros(len(pc), dtype=[
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('intensity', np.float32),
    ])
    data['x'] = pc[:, 0]
    data['y'] = pc[:, 1]
    data['z'] = pc[:, 2]
    if pc.shape[1] == 4:
        data['intensity'] = pc[:, 3]
    msg = ros_numpy.msgify(PointCloud2, data)
    msg.header = header
    publisher.publish(msg)


def crop_global_map_in_FOV(global_map, pose_estimation, cur_odom):
    # 当前scan原点的位姿
    print("裁剪global map")
    # print("cur_odom")
    # print(cur_odom)
    # defensive: ensure cur_odom is present and has header
    if cur_odom is None:
        if node:
            node.get_logger().warning('crop_global_map_in_FOV: cur_odom is None, returning empty submap')
        empty_pc = o3d.geometry.PointCloud()
        empty_pc.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        return empty_pc

    # pose_to_mat will convert Odometry->matrix
    T_odom_to_base_link = pose_to_mat(cur_odom)
    print("T_odom_to_base_link")
    print(T_odom_to_base_link)
    print("pose_estimation")
    print(pose_estimation)
    T_map_to_base_link = np.matmul(pose_estimation, T_odom_to_base_link)
    print("T_map_to_base_link")
    print(T_map_to_base_link)
    T_base_link_to_map = inverse_se3(T_map_to_base_link)
    print("T_base_link_to_map")
    print(T_base_link_to_map)

    # 把地图转换到lidar系下
    global_map_in_map = np.array(global_map.points)
    global_map_in_map = np.column_stack([global_map_in_map, np.ones(len(global_map_in_map))])
    global_map_in_base_link = np.matmul(T_base_link_to_map, global_map_in_map.T).T

    # 将视角内的地图点提取出来
    if FOV > 3.14:
        # 环状lidar 仅过滤距离
        indices = np.where(
            (np.abs(global_map_in_base_link[:, 0]) < FOV_FAR) & 
            (np.abs(global_map_in_base_link[:, 1]) < FOV_FAR)
            #&
            #(np.abs(np.arctan2(global_map_in_base_link[:, 1], global_map_in_base_link[:, 0])) < FOV / 2.0)
        )
    else:
        # 非环状lidar 保前视范围
        # FOV_FAR>x>0 且角度小于FOV
        indices = np.where(
            (global_map_in_base_link[:, 0] > 0) &
            (global_map_in_base_link[:, 0] < FOV_FAR) &
            (np.abs(np.arctan2(global_map_in_base_link[:, 1], global_map_in_base_link[:, 0])) < FOV / 2.0)
        )
    global_map_in_FOV = o3d.geometry.PointCloud()
    global_map_in_FOV.points = o3d.utility.Vector3dVector(np.squeeze(global_map_in_map[indices, :3]))

    # 发布fov内点云
    header = cur_odom.header
    header.frame_id = 'map'
    publish_point_cloud(pub_submap, header, np.array(global_map_in_FOV.points)[::10])

    return global_map_in_FOV


def global_localization(pose_estimation):
    global global_map, cur_scan, cur_odom, T_map_to_odom
    # 用icp配准
    # print(global_map, cur_scan, T_map_to_odom)
    if node:
        node.get_logger().info('Global localization by scan-to-map matching......')

    # TODO 这里注意线程安全
    scan_tobe_mapped = copy.copy(cur_scan)

    tic = time.time()

    global_map_in_FOV = crop_global_map_in_FOV(global_map, pose_estimation, cur_odom)

    # 粗配准
    print("粗配准...............")
    print("scan_tobe_mapped")
    print(scan_tobe_mapped)
    print("global_map_in_FOV")
    print(global_map_in_FOV)
    print("pose_estimation")
    print(pose_estimation)
    # Debug prints: show global_map, pose_estimation and cur_odom
    print("DEBUG: global_map info")
    try:
        gm_pts = np.array(global_map.points)
        print("global_map number of points:", gm_pts.shape[0])
        print("global_map first 5 points:\n", gm_pts[:5])
    except Exception as e:
        print("Could not print global_map:", e)

    print("DEBUG: pose_estimation")
    try:
        print(pose_estimation)
    except Exception as e:
        print("Could not print pose_estimation:", e)

    print("DEBUG: cur_odom")
    try:
        print(cur_odom)
    except Exception as e:
        print("Could not print cur_odom:", e)
    transformation, _ = registration_at_scale(scan_tobe_mapped, global_map_in_FOV, initial=pose_estimation, scale=5)
    #scan_tobe_mapped来自/cloud_registered，是将采集的雷达数据转换到odom坐标系
    #
    print("transformation")
    print(transformation)
    
    # 精配准
    transformation, fitness = registration_at_scale(scan_tobe_mapped, global_map_in_FOV, initial=transformation, scale=1)
    print("精配准..............")
    print("transformation")
    print(transformation)
    toc = time.time()
    print("匹配中......................")
    if node:
        node.get_logger().info('Time: {}'.format(toc - tic))
        node.get_logger().info('')
    print("fitness值为")
    print(fitness)
    # 当全局定位成功时才更新map2odom
    if fitness > LOCALIZATION_TH:
        # T_map_to_odom = np.matmul(transformation, pose_estimation)
        T_map_to_odom = transformation

        # if(T_map_to_odom_last is None):
        #     T_map_to_odom_last = T_map_to_odom
        # if(abs(T_map_to_odom[0][3] - T_map_to_odom_last[0][3]) > 0.1 and abs(T_map_to_odom[1][3] - T_map_to_odom_last[1][3]) > 0.1):
        #     T_map_to_odom = T_map_to_odom_last
        #     print("*******采用前一个数据************")
        # T_map_to_odom_last = T_map_to_odom
        
        # 发布map_to_odom
        map_to_odom = Odometry()
        xyz = tf_transformations.translation_from_matrix(T_map_to_odom)
        quat = tf_transformations.quaternion_from_matrix(T_map_to_odom)
        map_to_odom.pose.pose = Pose(Point(*xyz), Quaternion(*quat))
        # map_to_odom.header.stamp = cur_odom.header.stamp
        map_to_odom.header.stamp = lidar_time
        map_to_odom.header.frame_id = 'map'
        pub_map_to_odom.publish(map_to_odom)
        return True
    else:
        if node:
            node.get_logger().warning('Not match!!!!')
            node.get_logger().warning('{}' .format(transformation))
            node.get_logger().warning('fitness score:{}'.format(fitness))
        return False


def voxel_down_sample(pcd, voxel_size):
    try:
        pcd_down = pcd.voxel_down_sample(voxel_size)
    except:
        # for opend3d 0.7 or lower
        pcd_down = o3d.geometry.voxel_down_sample(pcd, voxel_size)
    return pcd_down


def initialize_global_map(pc_msg):
    global global_map

    global_map = o3d.geometry.PointCloud()
    global_map.points = o3d.utility.Vector3dVector(msg_to_array(pc_msg)[:, :3])
    global_map = voxel_down_sample(global_map, MAP_VOXEL_SIZE)
    if node:
        node.get_logger().info('Global map received.')


def cb_save_cur_odom(odom_msg):
    global cur_odom
    cur_odom = odom_msg


def cb_save_cur_scan(pc_msg):
    global cur_scan
    global lidar_time
    lidar_time = pc_msg.header.stamp
    # 注意这里fastlio直接将scan转到odom系下了 不是lidar局部系
    pc_msg.header.frame_id = 'camera_init'
    # pc_msg.header.stamp = rospy.Time().now()
    pub_pc_in_map.publish(pc_msg)

    # 转换为pcd
    # fastlio给的field有问题 处理一下
    pc_msg.fields = [pc_msg.fields[0], pc_msg.fields[1], pc_msg.fields[2],
                     pc_msg.fields[4], pc_msg.fields[5], pc_msg.fields[6],
                     pc_msg.fields[3], pc_msg.fields[7]]
    pc = msg_to_array(pc_msg)

    cur_scan = o3d.geometry.PointCloud()
    cur_scan.points = o3d.utility.Vector3dVector(pc[:, :3])


def thread_localization():
    global T_map_to_odom
    while True:
        # 每隔一段时间进行全局定位
        time.sleep(1 / FREQ_LOCALIZATION)
        # TODO 由于这里Fast lio发布的scan是已经转换到odom系下了 所以每次全局定位的初始解就是上一次的map2odom 不需要再拿odom了
        global_localization(T_map_to_odom)


if __name__ == '__main__':
    global lidar_time, pub_pc_in_map, pub_submap, pub_map_to_odom, node
    MAP_VOXEL_SIZE = 0.1
    SCAN_VOXEL_SIZE = 0.1

    # Global localization frequency (HZ)
    FREQ_LOCALIZATION = 0.5

    # The threshold of global localization,
    # only those scan2map-matching with higher fitness than LOCALIZATION_TH will be taken
    LOCALIZATION_TH = 0.91

    # FOV(rad), modify this according to your LiDAR type
    FOV = 3.15

    # The farthest distance(meters) within FOV
    FOV_FAR = 30

    rclpy.init()
    node = Node('fast_lio_localization')
    node.get_logger().info('Localization Node Inited...')

    qos = QoSProfile(depth=10)
    # publisher
    pub_pc_in_map = node.create_publisher(PointCloud2, '/cur_scan_in_map', qos)
    pub_submap = node.create_publisher(PointCloud2, '/submap', qos)
    pub_map_to_odom = node.create_publisher(Odometry, '/map_to_odom', qos)

    # subscribers
    node.create_subscription(PointCloud2, '/cloud_registered', lambda msg: cb_save_cur_scan(msg), qos)
    node.create_subscription(Odometry, '/t265/odom/sample', lambda msg: cb_save_cur_odom(msg), qos)

    # helper to wait for a single message on a topic
    def wait_for_message(topic, msg_type, timeout=None):
        container = {'msg': None}
        evt = threading.Event()

        def _cb(m):
            container['msg'] = m
            evt.set()

        sub = node.create_subscription(msg_type, topic, _cb, qos)
        got = evt.wait(timeout)
        node.destroy_subscription(sub)
        if got:
            return container['msg']
        return None

    # 初始化全局地图
    node.get_logger().warning('Waiting for global map......')
    map_msg = wait_for_message('/map', PointCloud2, timeout=None)
    if map_msg is None:
        node.get_logger().error('Did not receive global map')
        rclpy.shutdown()
        raise SystemExit(1)
    initialize_global_map(map_msg)

    # 初始化
    while not initialized:
        print("等待初始位姿..........")
        node.get_logger().warning('Waiting for initial pose....')

        # 等待初始位姿
        pose_msg = wait_for_message('/initialpose', PoseWithCovarianceStamped, timeout=None)
        if pose_msg is None:
            node.get_logger().warning('Initial pose wait timed out')
            time.sleep(0.1)
            continue
        initial_pose = pose_to_mat(pose_msg)
        print("initial_pose变换矩阵")
        print(initial_pose)
        if cur_scan:
            initialized = global_localization(initial_pose)
        else:
            node.get_logger().warning('First scan not received!!!!!')
        print(initialized)
        print("未接受到初始位姿..........")
    node.get_logger().info('')
    node.get_logger().info('Initialize successfully!!!!!!')
    node.get_logger().info('')
    # 开始定期全局定位
    threading.Thread(target=thread_localization, daemon=True).start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
