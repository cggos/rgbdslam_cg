find_package(PCL REQUIRED COMPONENTS commom io filters visualization)

find_package(OpenCV REQUIRED)

add_definitions(${PCL_DEFINITIONS})

include_directories(${PCL_INCLUDE_DIRS})

link_directories(${PCL_LIBRARY_DIRS})

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

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

add_library(slam_base slam_base.cpp)
target_link_libraries(slam_base ${OpenCV_LIBS} ${PCL_LIBRARIES})

add_executable(generate_pointcloud generate_pointcloud.cpp)
target_link_libraries(generate_pointcloud slam_base)

add_executable(detect_features detect_features.cpp)
target_link_libraries(detect_features slam_base)

add_executable(join_pointcloud join_pointcloud.cpp)
target_link_libraries(join_pointcloud slam_base)

add_executable(visual_odometry visual_odometry.cpp)
target_link_libraries(visual_odometry slam_base ${G2O_LIBRARIES})

add_executable(slam slam.cpp)
target_link_libraries(slam slam_base ${G2O_LIBRARIES})
