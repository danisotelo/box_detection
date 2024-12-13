cmake_minimum_required(VERSION 3.5.0)

# Project name and settings
project(BoxSegmentation)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find packages
find_package(OpenCV REQUIRED)
find_package(PCL 1.13 REQUIRED)
find_package(Eigen3 REQUIRED)

# Add the executable with multiple source files
add_executable(box_segmentation
    src/cpp/main.cpp
    src/cpp/core/BoxSegmenter.cpp
    src/cpp/core/preprocessor/Preprocessor.cpp
    src/cpp/core/detector/BoxDetector.cpp
    src/cpp/core/box/Box.cpp
    src/cpp/core/cluster/Cluster.cpp
    src/cpp/core/cluster/ClustersProcessor.cpp
    src/cpp/utils/utils.cpp
)

# Include directories for headers
target_include_directories(box_segmentation PRIVATE
    src/cpp
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
)

# Link OpenCV libraries
target_link_libraries(box_segmentation 
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES})

# Add definitions for PCL
add_definitions(${PCL_DEFINITIONS})
