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
	
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin) 
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib) 

link_directories(${PROJECT_SOURCE_DIR}/lib)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory(${PROJECT_SOURCE_DIR}/src)

add_executable(rgbdslam main.cpp)
