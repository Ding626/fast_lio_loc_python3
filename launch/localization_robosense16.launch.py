"""
ROS 2 launch file for Robosense RS-LiDAR-16
Replicates the original ROS1 launch behavior and arguments.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from pathlib import Path


def generate_launch_description():
    ld = LaunchDescription()

    try:
        pkg_share = get_package_share_directory('fast_lio_localization')
    except Exception:
        # fallback when package discovery isn't available (e.g. running from source)
        pkg_share = Path('/home/agilex/cag/SLAM_ws/install/fast_lio_localization/share/fast_lio_localization')
    config_yaml = Path('/home/agilex/cag/SLAM_ws/src/fast_lio_localization/config/robosense16.yaml')
    params_file = str(config_yaml) if config_yaml.exists() else None
    map_default = str(Path('/home/agilex/cag/SLAM_ws/src/fast_lio_localization/map/map.pcd'))

    rviz_arg = DeclareLaunchArgument('rviz', default_value='true', description='Launch rviz')
    map_arg = DeclareLaunchArgument('map', default_value=map_default, description='Path to map PCD')

    node_params = {
        'feature_extract_enable': False,
        'point_filter_num': 4,
        'max_iteration': 3,
        'filter_size_surf': 0.5,
        'filter_size_map': 0.5,
        'cube_side_length': 1000.0,
        'runtime_pos_log_enable': False,
        'pcd_save_enable': False,
    }

    # fastlio_mapping is a C++ node compiled from ROS1 sources and expects
    # ROS2-style parameters. The original repo shipped ROS1 YAML files; they
    # are not directly compatible. Pass only the node_params dictionary here to
    # avoid rcl argument parsing errors. If you want to use the YAML, convert
    # it to ROS2 parameters format first.
    launch_fastlio = Node(
        package='fast_lio_localization',
        executable='fastlio_mapping',
        name='laserMapping',
        output='log',
        parameters=[node_params]
    )

    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_init_to_odom',
        output='screen',
        arguments=['0', '0', '0', '0', '0', '0', 'camera_init', 'odom']
    )

    py_global_localization = Node(
        package='fast_lio_localization',
        executable='global_localization.py',
        name='global_localization',
        output='screen'
    )

    py_transform_fusion = Node(
        package='fast_lio_localization',
        executable='transform_fusion.py',
        name='transform_fusion',
        output='screen'
    )

    py_odom_tf_pub = Node(
        package='fast_lio_localization',
        executable='odom_tf_publisher.py',
        name='odom_tf_publisher',
        output='screen'
    )

    py_laser_tf_pub = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_link_to_laser',
            arguments=['0.1', '0', '0.4', '0', '0', '0', 'base_link', 'laser']
            # 参数说明: x y z yaw pitch roll frame_id child_frame_id [1](@ref)
        )

    # Provide the mandatory 'file_name' parameter (pcl_ros expects statically
    # typed parameter 'file_name' to be set). We also pass frame_id and count.
    # pcd_node = Node(
    #     package='pcl_ros',
    #     executable='pcd_to_pointcloud',
    #     name='map_publisher',
    #     output='screen',
    #     arguments=[LaunchConfiguration('map'), '5'],
    #     remappings=[('cloud_pcd', '/map')],
    #     parameters=[
    #         {'frame_id': '/map'},
    #         {'file_name': LaunchConfiguration('map')},
    #         {'count': 5}
    #     ]
    # )

    # Optional Python-based PCD publisher (uses Open3D and transient_local QoS)
    py_map_publisher = Node(
        package='fast_lio_localization',
        executable='publish_map_from_pcd.py',
        name='publish_map_from_pcd',
        output='screen',
        arguments=[LaunchConfiguration('map')],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        output='screen',
        arguments=['-d', str(Path(pkg_share) / 'rviz_cfg' / 'localization.rviz')],
        condition=IfCondition(LaunchConfiguration('rviz'))
    )

    ld.add_action(rviz_arg)
    ld.add_action(map_arg)
    ld.add_action(launch_fastlio)
    ld.add_action(static_tf)
    ld.add_action(py_global_localization)
    ld.add_action(py_transform_fusion)
    ld.add_action(py_odom_tf_pub)
    ld.add_action(py_laser_tf_pub)
    # ld.add_action(pcd_node)
    ld.add_action(py_map_publisher)
    ld.add_action(rviz_node)

    return ld