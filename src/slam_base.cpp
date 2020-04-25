/**
 * @file slam_base.cpp
 * @author cggos (cggos@outlook.com)
 * @brief 
 * @version 0.1
 * @date 2020-04-25
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core/eigen.hpp>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>

#include "slam_base.h"

PointCloud::Ptr image_to_pointcloud(const cv::Mat& rgb, const cv::Mat& depth, const CAMERA_INTRINSIC_PARAMETERS& camera) {
    PointCloud::Ptr cloud(new PointCloud);

    for (int m = 0; m < depth.rows; m++) {
        for (int n = 0; n < depth.cols; n++) {
            ushort d = depth.ptr<ushort>(m)[n];

            if (d == 0) continue;

            PointT p;

            p.z = double(d) / camera.scale;
            p.x = (n - camera.cx) * p.z / camera.fx;
            p.y = (m - camera.cy) * p.z / camera.fy;

            p.b = rgb.ptr<uchar>(m)[n * 3];
            p.g = rgb.ptr<uchar>(m)[n * 3 + 1];
            p.r = rgb.ptr<uchar>(m)[n * 3 + 2];

            cloud->points.push_back(p);
        }
    }
    cloud->height = 1;
    cloud->width = cloud->points.size();
    cloud->is_dense = false;

    return cloud;
}

cv::Point3f point2d_to_3d(const cv::Point3f& point, const CAMERA_INTRINSIC_PARAMETERS& camera) {
    cv::Point3f p;
    p.z = double(point.z) / camera.scale;
    p.x = (point.x - camera.cx) * p.z / camera.fx;
    p.y = (point.y - camera.cy) * p.z / camera.fy;
    return p;
}

void compute_keypoints_desp(FRAME& frame) {
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    detector->detect(frame.rgb, frame.kp);
    descriptor->compute(frame.rgb, frame.kp, frame.desp);
    return;
}

Eigen::Isometry3d cvmat_to_eigen(cv::Mat& rvec, cv::Mat& tvec) {
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    Eigen::Matrix3d r;
    cv::cv2eigen(R, r);

    // 将平移向量和旋转矩阵转换成变换矩阵
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();

    Eigen::AngleAxisd angle(r);
    // Eigen::Translation<double, 3> trans(tvec.at<double>(0, 0), tvec.at<double>(0, 1), tvec.at<double>(0, 2));
    T = angle;
    T(0, 3) = tvec.at<double>(0, 0);
    T(1, 3) = tvec.at<double>(0, 1);
    T(2, 3) = tvec.at<double>(0, 2);
    return T;
}

RESULT_OF_PNP estimate_motion(FRAME& frame1, FRAME& frame2, CAMERA_INTRINSIC_PARAMETERS& camera) {
    vector<cv::DMatch> matches;
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    matcher->match(frame1.desp, frame2.desp, matches);

    cout << "find total " << matches.size() << " matches." << endl;

    double min_dist = 10000, max_dist = 0;
    for (int i = 0; i < frame1.desp.rows; i++) {
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }
    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);
    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    std::vector<cv::DMatch> goodMatches;
    for (int i = 0; i < frame1.desp.rows; i++) {
        if (matches[i].distance <= max(2 * min_dist, 30.0))
            goodMatches.push_back(matches[i]);
    }

    cout << "good matches: " << goodMatches.size() << endl;

    vector<cv::Point3f> pts_obj;  // 第一个帧的三维点
    vector<cv::Point2f> pts_img;  // 第二个帧的图像点

    for (size_t i = 0; i < goodMatches.size(); i++) {
        cv::Point2f p = frame1.kp[goodMatches[i].queryIdx].pt;    // query 是第一个, train 是第二个
        ushort d = frame1.depth.ptr<ushort>(int(p.y))[int(p.x)];  // 获取d是要小心！x是向右的，y是向下的，所以y才是行，x是列！
        if (d == 0) continue;
        cv::Point2f pt2 = frame2.kp[goodMatches[i].trainIdx].pt;
        pts_img.push_back(pt2);
        cv::Point3f pt3 = point2d_to_3d(cv::Point3f(p.x, p.y, d), camera);  // 将(u,v,d)转成(x,y,z)
        pts_obj.push_back(pt3);
    }

    RESULT_OF_PNP result;

    if(pts_obj.size() < 4) {
        result.inliers = 0;
        return result;
    }

    cout << "solving pnp" << endl;
    double camera_matrix_data[3][3] = {
        {camera.fx, 0, camera.cx},
        {0, camera.fy, camera.cy},
        {0, 0, 1}};
    cv::Mat cameraMatrix(3, 3, CV_64F, camera_matrix_data);
    cv::Mat rvec, tvec, inliers;
    cv::solvePnPRansac(pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 8.0, 0.99, inliers);

    result.rvec = rvec;
    result.tvec = tvec;
    result.inliers = inliers.rows;

    return result;
}

PointCloud::Ptr join_pointcloud(PointCloud::Ptr original, FRAME& newFrame, Eigen::Isometry3d T, CAMERA_INTRINSIC_PARAMETERS& camera) {
    PointCloud::Ptr newCloud = image_to_pointcloud(newFrame.rgb, newFrame.depth, camera);

    // 合并点云
    PointCloud::Ptr output(new PointCloud());
    pcl::transformPointCloud(*original, *output, T.matrix());
    *newCloud += *output;

    // Voxel grid 滤波降采样
    static pcl::VoxelGrid<PointT> voxel;
    static ParameterReader pd;
    double gridsize = atof(pd.getData("voxel_grid").c_str());
    voxel.setLeafSize(gridsize, gridsize, gridsize);
    voxel.setInputCloud(newCloud);
    PointCloud::Ptr tmp(new PointCloud());
    voxel.filter(*tmp);
    return tmp;
}
