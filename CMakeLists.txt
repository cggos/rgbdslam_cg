cmake_minimum_required(VERSION 2.8) 

project(rgbdslam)

set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")
set(CMAKE_CXX_STANDARD 11)

set(TUM_SRC ${CMAKE_CURRENT_SOURCE_DIR}/include/tum_data_rgbd.h)
if(NOT EXISTS ${TUM_SRC})
    file(DOWNLOAD
            https://raw.githubusercontent.com/cggos/slam_park_cg/master/slam_dataset/tum/tum_data_rgbd.h
            ${TUM_SRC}
            SHOW_PROGRESS)
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(Eigen3 REQUIRED)
if(Eigen3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()

find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    # ${OpenCV_LIBS}
endif()

find_package(PCL REQUIRED COMPONENTS commom io filters visualization)
if(PCL_FOUND)
    add_definitions(${PCL_DEFINITIONS})
    include_directories(${PCL_INCLUDE_DIRS})
    link_directories(${PCL_LIBRARY_DIRS})
endif()

find_package(CSparse REQUIRED)
if(CSPARSE_FOUND)
    message("== CSPARSE_FOUND")
    include_directories(${CSPARSE_INCLUDE_DIR})
    # ${CSPARSE_LIBRARY}
endif()

find_package(Cholmod REQUIRED)
if(CHOLMOD_FOUND)
    message("== CHOLMOD_FOUND")
    include_directories(${CHOLMOD_INCLUDE_DIR})
    # ${CHOLMOD_LIBRARIES}
endif()

find_package(G2O REQUIRED)
if(G2O_FOUND)
    message("== G2O_FOUND")
    include_directories(${G2O_INCLUDE_DIR})
    link_directories($ENV{G2O_ROOT}/lib)
    set(G2O_LIBRARIES g2o_core g2o_stuff g2o_types_sba g2o_types_slam3d g2o_types_sim3 g2o_solver_cholmod g2o_solver_csparse g2o_csparse_extension)
endif()
	
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin) 
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib) 

link_directories(${PROJECT_SOURCE_DIR}/lib)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory(${PROJECT_SOURCE_DIR}/src)

add_executable(generate_pointcloud app/generate_pointcloud.cpp)
target_link_libraries(generate_pointcloud slam_base)

add_executable(detect_features app/detect_features.cpp)
target_link_libraries(detect_features slam_base)

add_executable(join_pointcloud app/join_pointcloud.cpp)
target_link_libraries(join_pointcloud slam_base)

add_executable(visual_odometry app/visual_odometry.cpp)
target_link_libraries(visual_odometry slam_base ${G2O_LIBRARIES})

add_executable(rgbdslam app/slam.cpp)
target_link_libraries(rgbdslam slam_base ${G2O_LIBRARIES})
