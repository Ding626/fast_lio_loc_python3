cmake_minimum_required(VERSION 3.5)
project(fast_lio_localization)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters)
find_package(Python3 COMPONENTS Interpreter Development)

# Generate ROS interfaces from msg/ files
find_package(rosidl_default_generators REQUIRED)
# find_package(fast_lio_localization REQUIRED)
# Generate message headers (must be declared before building targets that use them)
# Skip the group membership check here to avoid ordering issues with ament's
# package.xml parsing during configuration. package.xml already contains
# the required <member_of_group>rosidl_interface_packages</member_of_group> export.
rosidl_generate_interfaces(${PROJECT_NAME}
  SKIP_GROUP_MEMBERSHIP_CHECK
  "msg/Pose6D.msg"
)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)

# ikd-Tree submodule is expected under include/ikd-Tree
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/ikd-Tree/ikd_Tree.h")
  message(STATUS "Found ikd-Tree headers; adding include directory")
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
else()
  message(FATAL_ERROR "ikd-Tree headers not found. Run: git submodule update --init --recursive\nor install ikd-Tree into include/ikd-Tree so that ikd-Tree/ikd_Tree.h exists.")
endif()

# Install Python scripts so they can be run with ros2 run <pkg> <executable>
file(GLOB PY_SCRIPTS "${CMAKE_CURRENT_SOURCE_DIR}/scripts/*.py")
install(PROGRAMS ${PY_SCRIPTS}
  DESTINATION lib/${PROJECT_NAME}
)

# NOTE: The C++ sources in src/ use ROS1 APIs and must be ported to rclcpp.
# The build target for the original C++ node is left commented as a reminder
# and should be updated after porting the code to ROS 2 (rclcpp, tf2_ros, etc.).
# Add the migrated C++ node target. Sources may depend on other headers in include/.
add_executable(fastlio_mapping
  src/laserMapping.cpp
  src/preprocess.cpp
  include/ikd-Tree/ikd_Tree.cpp
)

target_include_directories(fastlio_mapping PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# Ensure generated message headers are available to the target
# Add the rosidl generator output parent directory so includes like
# <fast_lio_localization/msg/pose6_d.hpp> resolve to
# ${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_cpp/fast_lio_localization/msg/pose6_d.hpp
target_include_directories(fastlio_mapping PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_cpp>
)

# Make sure the target depends on message generation
# Use the target variable set by rosidl_generate_interfaces to ensure the
# build target for generated code is created before compiling the C++ target.
add_dependencies(fastlio_mapping ${rosidl_generate_interfaces_TARGET})

# Link against PCL and ament packages
ament_target_dependencies(fastlio_mapping
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  tf2
  tf2_ros
  pcl_conversions
  visualization_msgs
  tf2_geometry_msgs
)

target_link_libraries(fastlio_mapping
  ${PCL_LIBRARIES}
  Python3::Python
)

install(TARGETS fastlio_mapping
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY launch config map msg scripts
  DESTINATION share/${PROJECT_NAME}
)
# Minimal skeleton node to validate ament build and provide a porting stub
# add_executable(laser_mapping_skeleton src/laser_mapping_node.cpp)
# ament_target_dependencies(laser_mapping_skeleton rclcpp visualization_msgs)
# install(TARGETS laser_mapping_skeleton
#   RUNTIME DESTINATION lib/${PROJECT_NAME}
# )

ament_package()
