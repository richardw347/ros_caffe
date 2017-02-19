cmake_minimum_required(VERSION 2.8.3)
project(ros_caffe)

add_definitions(-DCPU_ONLY=1)

find_package(catkin REQUIRED COMPONENTS
  roscpp roslib std_msgs sensor_msgs image_transport cv_bridge spark_msgs
)

set(CAFFE_INCLUDEDIR caffe/include caffe/distribute/include)
set(CAFFE_LINK_LIBRARAY caffe/build/lib)

find_package(Boost REQUIRED COMPONENTS system)
find_package(OpenCV REQUIRED)

catkin_package(
	CATKIN_DEPENDS
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  /usr/include/hdf5/serial/
)
include_directories(${CAFFE_INCLUDEDIR})

link_directories(
  ${CAFFE_LINK_LIBRARAY}
  /usr/lib/x86_64-linux-gnu/hdf5/serial/
)

add_executable(ros_caffe_test src/ros_caffe_test.cpp)
target_link_libraries(ros_caffe_test ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} caffe glog)
