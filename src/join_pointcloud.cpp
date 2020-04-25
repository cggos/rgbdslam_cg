/**
 * @file join_pointcloud.cpp
 * @author cggos (cggos@outlook.com)
 * @brief 
 * @version 0.1
 * @date 2020-04-25
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#include <iostream>
using namespace std;

#include <opencv2/calib3d/calib3d.hpp>

#include <pcl/visualization/cloud_viewer.h>

#include "slam_base.h"

int main(int argc, char** argv) {
    ParameterReader pd;
    FRAME frame1, frame2;

    frame1.rgb = cv::imread("../data/rgb1.png");
    frame1.depth = cv::imread("../data/depth1.png", -1);
    frame2.rgb = cv::imread("../data/rgb2.png");
    frame2.depth = cv::imread("../data/depth2.png", -1);

    cout << "extracting features" << endl;
    compute_keypoints_desp(frame1);
    compute_keypoints_desp(frame2);

    // 相机内参
    CAMERA_INTRINSIC_PARAMETERS camera;
    camera.fx = atof(pd.getData("camera.fx").c_str());
    camera.fy = atof(pd.getData("camera.fy").c_str());
    camera.cx = atof(pd.getData("camera.cx").c_str());
    camera.cy = atof(pd.getData("camera.cy").c_str());
    camera.scale = atof(pd.getData("camera.scale").c_str());

    RESULT_OF_PNP result = estimate_motion(frame1, frame2, camera);

    cout << "rvec = " << result.rvec.t() << endl;
    cout << "tvec = " << result.tvec.t() << endl;

    // 将平移向量和旋转矩阵转换成变换矩阵
    Eigen::Isometry3d T = cvmat_to_eigen(result.rvec, result.tvec);

    cout << "converting image to clouds" << endl;

    PointCloud::Ptr cloud1 = image_to_pointcloud(frame1.rgb, frame1.depth, camera);

    PointCloud::Ptr output = join_pointcloud(cloud1, frame2, T, camera);

    pcl::io::savePCDFile("result.pcd", *output);
    cout << "Final result saved." << endl;

    pcl::visualization::CloudViewer viewer("viewer");
    viewer.showCloud(output);
    while (!viewer.wasStopped()) {}
    
    return 0;
}