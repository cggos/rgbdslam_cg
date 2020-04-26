

add_library(slam_base slam_base.cpp)
target_link_libraries(slam_base ${OpenCV_LIBS} ${PCL_LIBRARIES})
