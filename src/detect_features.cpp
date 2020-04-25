/*************************************************************************
	> File Name: detectFeatures.cpp
	> Author: xiang gao
	> Mail: gaoxiang12@mails.tsinghua.edu.cn
    > 特征提取与匹配
	> Created Time: 2015年07月18日 星期六 16时00分21秒
 ************************************************************************/

#include <iostream>
using namespace std;

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "slam_base.h"

int main(int argc, char** argv) {
    cv::Mat rgb1 = cv::imread("../data/rgb1.png");
    cv::Mat rgb2 = cv::imread("../data/rgb2.png");
    cv::Mat depth1 = cv::imread("../data/depth1.png", -1);
    cv::Mat depth2 = cv::imread("../data/depth2.png", -1);

    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();

    vector<cv::KeyPoint> kp1, kp2;  //关键点
    detector->detect(rgb1, kp1);    //提取关键点
    detector->detect(rgb2, kp2);

    cout << "Key points of two images: " << kp1.size() << ", " << kp2.size() << endl;

    // 可视化， 显示关键点
    cv::Mat imgShow;
    cv::drawKeypoints(rgb1, kp1, imgShow, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow("keypoints", imgShow);
    cv::imwrite("./data/keypoints.png", imgShow);
    cv::waitKey(0);  //暂停等待一个按键

    // 计算描述子
    cv::Mat desp1, desp2;
    descriptor->compute(rgb1, kp1, desp1);
    descriptor->compute(rgb2, kp2, desp2);

    // 匹配描述子
    std::vector<cv::DMatch> matches;
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create( "BruteForce-Hamming" );
    matcher->match(desp1, desp2, matches);
    cout << "Find total " << matches.size() << " matches." << endl;

    // 可视化：显示匹配的特征
    cv::Mat imgMatches;
    cv::drawMatches(rgb1, kp1, rgb2, kp2, matches, imgMatches);
    cv::imshow("matches", imgMatches);
    cv::imwrite("matches.png", imgMatches);
    cv::waitKey(0);

    double min_dist=10000, max_dist=0;
    for ( int i = 0; i < desp1.rows; i++ ) {
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }
    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );
    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    std::vector<cv::DMatch> goodMatches;
    for ( int i = 0; i < desp1.rows; i++ ) {
        if (matches[i].distance <= max(2 * min_dist, 30.0))
            goodMatches.push_back(matches[i]);
    }

    // 显示 good matches
    cout << "good matches=" << goodMatches.size() << endl;
    cv::drawMatches(rgb1, kp1, rgb2, kp2, goodMatches, imgMatches);
    cv::imshow("good matches", imgMatches);
    cv::imwrite("good_matches.png", imgMatches);
    cv::waitKey(0);

    vector<cv::Point3f> pts_obj;  // 第一个帧的三维点
    vector<cv::Point2f> pts_img;  // 第二个帧的图像点

    // rgbd_dataset_freiburg2_desk dataset
    CAMERA_INTRINSIC_PARAMETERS C;
    C.cx = 325.1;
    C.cy = 249.7;
    C.fx = 520.9;
    C.fy = 521.0;
    C.scale = 5000.0;

    for (size_t i = 0; i < goodMatches.size(); i++) {
        cv::Point2f p = kp1[goodMatches[i].queryIdx].pt;  // query 是第一个, train 是第二个

        ushort d = depth1.ptr<ushort>(int(p.y))[int(p.x)];  // 获取d是要小心！x是向右的，y是向下的，所以y才是行，x是列！

        if (d == 0) continue;

        cv::Point2f pt2 = kp2[goodMatches[i].trainIdx].pt;
        pts_img.push_back(pt2);

        cv::Point3f pt3 = point2d_to_3d(cv::Point3f(p.x, p.y, d), C); // 将(u,v,d)转成(x,y,z)
        pts_obj.push_back(pt3);

        char num[5];
        sprintf(num, "%03d", i);
        std::cout << std::string(num) << ": " << pt2 << ", " << pt3 << std::endl;
    }

    double camera_matrix_data[3][3] = {
        {C.fx, 0, C.cx},
        {0, C.fy, C.cy},
        {0, 0, 1}};

    cv::Mat cameraMatrix(3, 3, CV_64F, camera_matrix_data);
    cv::Mat rvec, tvec, inliers;
    cv::solvePnPRansac(pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 8.0, 0.99, inliers);
    // cv::solvePnP(pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, cv::SOLVEPNP_EPNP);

    cout << "rvec = " << rvec.t() << endl;
    cout << "tvec = " << tvec.t() << endl;
    cout << "inliers: " << inliers.rows << endl;

    // 画出inliers匹配
    vector<cv::DMatch> matchesShow;
    for (size_t i = 0; i < inliers.rows; i++) {
        matchesShow.push_back(goodMatches[inliers.ptr<int>(i)[0]]);
    }
    cv::drawMatches(rgb1, kp1, rgb2, kp2, matchesShow, imgMatches);
    cv::imshow("inlier matches", imgMatches);
    cv::imwrite("inliers.png", imgMatches);
    cv::waitKey(0);

    return 0;
}
