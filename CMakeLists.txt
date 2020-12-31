cmake_minimum_required(VERSION 3.9)
project(TF_OD_C_Plus_Plus)

set(CMAKE_CXX_STANDARD 11)
set(TF_ROOT "/home/chenwei/tensorflow-1.12.3")

find_package(PCL REQUIRED)
if (PCL_FOUND)
    link_directories(${PCL_LIBRARY_DIRS})
    add_definitions(${PCL_DEFINITIONS})
endif()

find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    link_directories(${OpenCV_LIBRARY_DIRS})
    add_definitions(${OpenCV_DEFINITIONS})
endif ()

find_package(catkin REQUIRED COMPONENTS
    roscpp
    rosbag
    std_msgs
    sensor_msgs
    pcl_ros
)

catkin_package(
        INCLUDE_DIRS include
        CATKIN_DEPENDS roscpp rosbag std_msgs sensor_msgs pcl_ros
)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${TF_ROOT}/bazel-genfiles
    ${TF_ROOT}/bazel-tensorflow-1.12.3
    ${TF_ROOT}/bazel-tensorflow-1.12.3/tensorflow/contrib/makefile/gen/protobuf/include
    ${TF_ROOT}/bazel-tensorflow-1.12.3/tensorflow/contrib/makefile/gen/host_obj
    ${TF_ROOT}/bazel-tensorflow-1.12.3/tensorflow/contrib/makefile/gen/proto
    ${TF_ROOT}/bazel-tensorflow-1.12.3/tensorflow/contrib/makefile/downloads/nsync/public
    ${TF_ROOT}/bazel-tensorflow-1.12.3/tensorflow/contrib/makefile/downloads/eigen
    ${TF_ROOT}/bazel-tensorflow-1.12.3/bazel-out/host/genfiles
    ${TF_ROOT}/bazel-tensorflow-1.12.3/tensorflow/contrib/makefile/downloads/absl
)

add_executable(TF_OD_C_Plus_Plus complex-yolo.cpp src/inference.cpp src/postprocess.cpp src/preprocess.cpp)

target_link_libraries(TF_OD_C_Plus_Plus
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${PCL_LIBRARIES}
    ${TF_ROOT}/bazel-bin/tensorflow/libtensorflow_cc.so
    ${TF_ROOT}/bazel-bin/tensorflow/libtensorflow_framework.so
)