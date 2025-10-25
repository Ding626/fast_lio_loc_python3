#!/usr/bin/env python3
# coding=utf8
from __future__ import print_function, division, absolute_import
import tf2_ros
import copy
import _thread as thread
import time
import threading
import threading
import open3d as o3d
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
import ros2_numpy
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Point, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
import numpy as np

# small replacements for tf_transformations used in this file
def translation_matrix(t):
    T = np.eye(4, dtype=float)
    T[0:3, 3] = np.array(t, dtype=float)
    return T


def quaternion_matrix(q):
    # q: (x, y, z, w)
    x, y, z, w = q
    # normalize
    qn = np.array([x, y, z, w], dtype=float)
    qn = qn / np.linalg.norm(qn)
    x, y, z, w = qn
    R = np.array([
        [1 - 2 * (y * y + z * z),     2 * (x * y - z * w),     2 * (x * z + y * w)],
        [    2 * (x * y + z * w), 1 - 2 * (x * x + z * z),     2 * (y * z - x * w)],
        [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]
    ], dtype=float)
    T = np.eye(4, dtype=float)
    T[0:3, 0:3] = R
    return T


def translation_from_matrix(T):
    return T[0:3, 3].copy()


def quaternion_from_matrix(T):
    R = T[0:3, 0:3]
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

    T_trans = translation_matrix((pos.x, pos.y, pos.z))
    T_rot = quaternion_matrix((ori.x, ori.y, ori.z, ori.w))
    return np.matmul(T_trans, T_rot)
  

def msg_to_array(pc_msg):
    """Convert a PointCloud2 message to a numpy array of points.
    This function is robust to different PointCloud2 representations.
    """
    # The primary method is to use sensor_msgs_py.point_cloud2.read_points
    # which is the most robust and official way to parse PointCloud2 messages.
    try:
        from sensor_msgs_py import point_cloud2
        # read_points returns an iterator of structured data.
        points_generator = point_cloud2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True)
        
        # Convert the generator to a list, then to a structured numpy array.
        points_list = list(points_generator)
        if not points_list:
            return np.zeros((0, 3), dtype=np.float64)
        
        # Create a structured array from the list of tuples
        structured_array = np.array(points_list, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        
        # Manually create a new float64 array and copy data column by column.
        # This is more robust than astype() for structured arrays.
        pc_array = np.zeros((len(structured_array), 3), dtype=np.float64)
        pc_array[:, 0] = structured_array['x']
        pc_array[:, 1] = structured_array['y']
        pc_array[:, 2] = structured_array['z']
        
        return pc_array

    except Exception as e:
        if node:
            node.get_logger().error(f'Failed to parse PointCloud2 to array using sensor_msgs_py: {e}')

    # Fallback to ros2_numpy if sensor_msgs_py fails for some reason.
    try:
        if node:
            node.get_logger().warning('Falling back to ros2_numpy to parse PointCloud2.')
        pc_array = ros2_numpy.numpify(pc_msg)
        
        if hasattr(pc_array, 'dtype') and pc_array.dtype.names:
            # It's a structured array. Extract x, y, z.
            x = pc_array['x']
            y = pc_array['y']
            z = pc_array['z']
            # Stack them and ensure the result is float64.
            return np.vstack([x, y, z]).T.astype(np.float64)
        elif isinstance(pc_array, np.ndarray) and pc_array.ndim == 2 and pc_array.shape[1] >= 3:
            # It's a regular numpy array.
            return pc_array[:, :3].astype(np.float64)

    except Exception as e_ros2numpy:
        if node:
            node.get_logger().error(f'Fallback to ros2_numpy also failed: {e_ros2numpy}')

    # If all methods fail, return an empty array.
    return np.zeros((0, 3), dtype=np.float64)


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
    # Try ros2_numpy first, but some ros2_numpy versions expect a dict for
    # point cloud input. Fall back to sensor_msgs_py.point_cloud2.create_cloud
    # if msgify raises an exception.
    try:
        msg = ros2_numpy.msgify(PointCloud2, data)
        msg.header = header
        publisher.publish(msg)
        return
    except Exception as e:
        if node:
            node.get_logger().warning('ros2_numpy.msgify failed, falling back to sensor_msgs_py: %s' % str(e))

    # Fallback: build PointCloud2 with sensor_msgs_py.point_cloud2.create_cloud
    try:
        from sensor_msgs_py import point_cloud2
        from sensor_msgs.msg import PointField
        # Build fields list
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        offset = 12
        if pc.shape[1] == 4:
            fields.append(PointField(name='intensity', offset=offset, datatype=PointField.FLOAT32, count=1))
            offset += 4

        # Prepare rows as list of tuples
        if pc.shape[1] == 4:
            rows = [tuple(map(float, p[:4])) for p in pc]
        else:
            rows = [tuple(map(float, p[:3])) for p in pc]

        msg = point_cloud2.create_cloud(header, fields, rows)
        publisher.publish(msg)
    except Exception as e:
        if node:
            node.get_logger().error('Failed to publish point cloud fallback: %s' % str(e))


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
    print("global_map_all")
    print(global_map)
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
        xyz = translation_from_matrix(T_map_to_odom)
        quat = quaternion_from_matrix(T_map_to_odom)
        # Correctly initialize the Point and Quaternion objects
        point = Point()
        point.x, point.y, point.z = xyz
        quaternion = Quaternion()
        quaternion.x, quaternion.y, quaternion.z, quaternion.w = quat
        map_to_odom.pose.pose = Pose(position=point, orientation=quaternion)
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
    pc = msg_to_array(pc_msg)
    global_map = o3d.geometry.PointCloud()
    global_map.points = o3d.utility.Vector3dVector(pc[:, :3])
    


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


def wait_for_message(topic, msg_type, timeout=None, qos_profile=None):
    container = {'msg': None}
    evt = threading.Event()
    def _cb(m):
        container['msg'] = m
        evt.set()
    # Use provided qos_profile if given; otherwise fall back to the module-level
    # `qos` if it exists, or construct a reasonable default. Passing None to
    # create_subscription raises a TypeError in rclpy, so ensure we always
    # provide a QoSProfile or an int depth.
    
    sub = node.create_subscription(msg_type, topic, _cb, qos_profile)
    start_time = time.monotonic()
    while not evt.is_set():
        rclpy.spin_once(node, timeout_sec=0.1)
        if timeout is not None and time.monotonic() - start_time > timeout:
            break
    node.destroy_subscription(sub)
    if evt.is_set():
        return container['msg']
    return None

if __name__ == '__main__':
    # module-level globals are declared where needed inside functions; do not use
    # a global statement at module scope (invalid syntax). The variables used
    # below will be assigned directly.
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
    node.create_subscription(Odometry, '/odom', lambda msg: cb_save_cur_odom(msg), qos)

    # helper to wait for a single message on a topic

    # 初始化全局地图
    node.get_logger().warning('Waiting for global map......')
    # Use TRANSIENT_LOCAL durability for map so late subscribers can receive
    # a previously-published map (common behavior for static maps).
    map_qos = QoSProfile(depth=1)
    map_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
    map_qos.reliability = QoSReliabilityPolicy.RELIABLE
    map_msg = wait_for_message('/map', PointCloud2, timeout=None, qos_profile=map_qos)
    if map_msg is None:
        node.get_logger().error('Did not receive global map')
        rclpy.shutdown()
        raise SystemExit(1)
    print("initialize_global_map(map_msg)")
    initialize_global_map(map_msg)

    # 初始化
    while not initialized:
        print("等待初始位姿..........")
        node.get_logger().warning('Waiting for initial pose....')

        # 等待初始位姿
        pose_qos = QoSProfile(depth=1)
        pose_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        pose_qos.reliability = QoSReliabilityPolicy.RELIABLE
        pose_msg = wait_for_message('/initialpose', PoseWithCovarianceStamped, timeout=None, qos_profile=pose_qos)
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
