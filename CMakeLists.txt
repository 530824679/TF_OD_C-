cmake_minimum_required(VERSION 3.9)
project(TF_OD_C__)

set(CMAKE_CXX_STANDARD 11)
set(TF_ROOT "/home/chenwei/tensorflow-1.12.3")

include_directories(
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

add_executable(TF_OD_C__ complex-yolo.cpp)

target_link_libraries(TF_OD_C__
    ${TF_ROOT}/bazel-bin/tensorflow/libtensorflow_cc.so
    ${TF_ROOT}/bazel-bin/tensorflow/libtensorflow_framework.so
)