/**
 * @file slam_base.h
 * @author cggos (cggos@outlook.com)
 * @brief rgbd-slam教程所用到的基本函数（C风格）
 * @version 0.1
 * @date 2020-04-25
 * 
 * @copyright Copyright (c) 2020
 * 
 */
#pragma once

#include <fstream>
#include <vector>
using namespace std;

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

struct CAMERA_INTRINSIC_PARAMETERS {
    double cx, cy, fx, fy, scale;
};

struct FRAME {
    cv::Mat rgb, depth;       //该帧对应的彩色图与深度图
    cv::Mat desp;             //特征描述子
    vector<cv::KeyPoint> kp;  //关键点
    FRAME() {}
    FRAME(cv::Mat rgb, cv::Mat depth) : rgb(rgb), depth(depth) {}
};

struct RESULT_OF_PNP {
    cv::Mat rvec, tvec;
    int inliers;
};

class ParameterReader {
   public:
    ParameterReader(string filename = "../config/params.txt") {
        std::ifstream fin(filename.c_str());
        if (!fin) {
            cerr << "parameter file does not exist." << endl;
            return;
        }
        while (!fin.eof()) {
            string str;
            getline(fin, str);
            if (str[0] == '#') {
                // 以‘＃’开头的是注释
                continue;
            }

            int pos = str.find("=");
            if (pos == -1)
                continue;
            string key = str.substr(0, pos);
            string value = str.substr(pos + 1, str.length());
            data[key] = value;

            if (!fin.good())
                break;
        }
    }
    string getData(string key) {
        map<string, string>::iterator iter = data.find(key);
        if (iter == data.end()) {
            cerr << "Parameter name " << key << " not found!" << endl;
            return string("NOT_FOUND");
        }
        return iter->second;
    }

   public:
    map<string, string> data;
};

/**
 * @brief 将rgb图转换为点云
 * 
 * @param rgb 
 * @param depth 
 * @param camera 
 * @return PointCloud::Ptr 
 */
PointCloud::Ptr image_to_pointcloud(const cv::Mat& rgb, const cv::Mat& depth, const CAMERA_INTRINSIC_PARAMETERS& camera);

/**
 * @brief 将单个点从图像坐标转换为空间坐标
 * 
 * @param point 3维点Point3f (u,v,d)
 * @param camera 
 * @return cv::Point3f 
 */
cv::Point3f point2d_to_3d(const cv::Point3f& point, const CAMERA_INTRINSIC_PARAMETERS& camera);

/**
 * @brief 同时提取关键点与特征描述子
 * 
 * @param frame 
 * @param detector 
 * @param descriptor 
 */
void compute_keypoints_desp(FRAME& frame);

/**
 * @brief convert OpenCV Mat (rvec,tvec) to Eigen Isometry3d
 * 
 * @param rvec 
 * @param tvec 
 * @return Eigen::Isometry3d 
 */
Eigen::Isometry3d cvmat_to_eigen(cv::Mat& rvec, cv::Mat& tvec);

/**
 * @brief 计算两个帧之间的运动
 * 
 * @param frame1 
 * @param frame2 
 * @param camera 相机内参
 * @return RESULT_OF_PNP 
 */
RESULT_OF_PNP estimate_motion(FRAME& frame1, FRAME& frame2, CAMERA_INTRINSIC_PARAMETERS& camera);

/**
 * @brief 
 * 
 * @param original 
 * @param newFrame 
 * @param T 
 * @param camera 
 * @return PointCloud::Ptr 
 */
PointCloud::Ptr join_pointcloud(PointCloud::Ptr original, FRAME& newFrame, Eigen::Isometry3d T, CAMERA_INTRINSIC_PARAMETERS& camera);
