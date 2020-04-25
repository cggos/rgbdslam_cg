/**
 * @file generate_pointcloud.cpp
 * @author cggos (cggos@outlook.com)
 * @brief 读取./data/rgb.png和./data/depth.png，并转化为点云
 * @version 0.1
 * @date 2020-04-25
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#include "slam_base.h"

int main(int argc, char** argv) {
    cv::Mat rgb, depth;
    rgb = cv::imread("../data/rgb1.png");
    depth = cv::imread("../data/depth1.png", -1);  // depth 是16UC1的单通道图像

    CAMERA_INTRINSIC_PARAMETERS C;
    C.cx = 325.1;
    C.cy = 249.7;
    C.fx = 520.9;
    C.fy = 521.0;
    C.scale = 5000.0;

    PointCloud::Ptr cloud(new PointCloud);
    cloud = image_to_pointcloud(rgb, depth, C);

    pcl::io::savePCDFile("./pointcloud.pcd", *cloud);

    cout << "Point cloud saved." << endl;

    return 0;
}