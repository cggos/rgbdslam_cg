/**
 * @file visual_odometry.cpp
 * @author cggos (cggos@outlook.com)
 * @brief 
 * @version 0.1
 * @date 2020-04-25
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#include <fstream>
#include <iostream>
#include <sstream>
using namespace std;

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/slam3d/edge_se3.h>
#include <pcl/visualization/cloud_viewer.h>

#include "slam_base.h"
#include "tum_data_rgbd.h"

int main(int argc, char** argv) {
    ParameterReader pd;
    int idx_s = atoi(pd.getData("start_index").c_str());
    int idx_e = atoi(pd.getData("end_index").c_str());
    bool visualize = pd.getData("visualize_pointcloud") == string("yes");
    int min_inliers = atoi(pd.getData("min_inliers").c_str());
    double max_norm = atof(pd.getData("max_norm").c_str());

    CAMERA_INTRINSIC_PARAMETERS camera;
    vector<cv::Mat> color_imgs, depth_imgs;
    cg::TUMDataRGBD tum_data_rgbd("/home/cg/dev_sdb/datasets/TUM/RGBD-SLAM-Dataset/rgbd_dataset_freiburg2_desk/", 1);
    {
        cv::Mat K;
        tum_data_rgbd.getK(K);
        camera.fx = K.at<double>(0, 0);
        camera.fy = K.at<double>(1, 1);
        camera.cx = K.at<double>(0, 2);
        camera.cy = K.at<double>(1, 2);
        camera.scale = tum_data_rgbd.depth_scale_;
    }

    int idx_c, idx_p;
    FRAME frame_c, frame_p;

    PointCloud::Ptr cloud;
    pcl::visualization::CloudViewer viewer("viewer");

    auto linearSolver = g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
    linearSolver->setBlockOrdering(false);
    auto blockSolver = g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver));
    g2o::OptimizationAlgorithmLevenberg* optimizationAlgorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));

    // std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver(new g2o::LinearSolverCSparse<g2o::BlockSolver_6_3::PoseMatrixType>());
    // std::unique_ptr<g2o::BlockSolver_6_3> solver_ptr(new g2o::BlockSolver_6_3(std::move(linearSolver)));
    // g2o::OptimizationAlgorithmLevenberg* optimizationAlgorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));

    g2o::SparseOptimizer globalOptimizer;
    globalOptimizer.setVerbose(true);
    globalOptimizer.setAlgorithm(optimizationAlgorithm);

    for (int i = 0; i < idx_e + 1; ++i) {
        cv::Mat img_color, img_depth;
        if (!tum_data_rgbd.get_rgb_depth(img_color, img_depth)) {
            std::cerr << "get_rgb failed!" << std::endl;
            return -1;
        }

        if (i < idx_s) continue;

        idx_c = i;

        if (idx_c == idx_s) {
            cout << "Initializing ..." << endl;
            
            frame_c = FRAME(img_color, img_depth);
            compute_keypoints_desp(frame_c);

            cloud = image_to_pointcloud(frame_c.rgb, frame_c.depth, camera);

            // 向globalOptimizer增加第一个顶点
            g2o::VertexSE3* v = new g2o::VertexSE3();
            v->setId(idx_c);
            v->setEstimate(Eigen::Isometry3d::Identity());  //估计为单位矩阵
            v->setFixed(true);                              //第一个顶点固定，不用优化
            globalOptimizer.addVertex(v);

            idx_p = idx_c;
            frame_p = frame_c;

            continue;
        }

        cout << "current index: " << idx_c << endl;
        frame_c = FRAME(img_color, img_depth);
        compute_keypoints_desp(frame_c);

        RESULT_OF_PNP result = estimate_motion(frame_p, frame_c, camera);

        std::cout << "inliers: " << result.inliers << std::endl;
        if (result.inliers < min_inliers) continue;

        // 计算运动范围是否太大
        double norm = fabs(min(cv::norm(result.rvec), 2 * M_PI - cv::norm(result.rvec))) + fabs(cv::norm(result.tvec));
        cout << "norm = " << norm << endl;
        if (norm >= max_norm) continue;

        Eigen::Isometry3d T = cvmat_to_eigen(result.rvec, result.tvec);

        if (visualize == true) {
            cloud = join_pointcloud(cloud, frame_c, T, camera);
            viewer.showCloud(cloud);
        }

        // 向g2o中增加这个顶点与上一帧联系的边
        // 顶点只需设定id即可
        g2o::VertexSE3* v = new g2o::VertexSE3();
        v->setId(idx_c);
        v->setEstimate(Eigen::Isometry3d::Identity());
        globalOptimizer.addVertex(v);

        g2o::EdgeSE3* edge = new g2o::EdgeSE3();
        edge->vertices()[0] = globalOptimizer.vertex(idx_p);
        edge->vertices()[1] = globalOptimizer.vertex(idx_c);
        Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
        // 信息矩阵是协方差矩阵的逆，表示我们对边的精度的预先估计
        // 因为pose为6D的，信息矩阵是6*6的阵，假设位置和角度的估计精度均为0.1且互相独立
        // 那么协方差则为对角为0.01的矩阵，信息阵则为100的矩阵
        // 也可以将角度设大一些，表示对角度的估计更加准确
        information(0, 0) = information(1, 1) = information(2, 2) = 100;
        information(3, 3) = information(4, 4) = information(5, 5) = 100;
        edge->setInformation(information);
        edge->setMeasurement(T);  // 边的估计即是pnp求解之结果
        globalOptimizer.addEdge(edge);

        idx_p = idx_c;
        frame_p = frame_c;

        usleep(30000);
    }

    // 优化所有边
    cout << "optimizing pose graph, vertices: " << globalOptimizer.vertices().size() << endl;
    globalOptimizer.save("result_before.g2o");
    globalOptimizer.initializeOptimization();
    globalOptimizer.optimize(100);  //可以指定优化步数
    globalOptimizer.save("result_after.g2o");

    globalOptimizer.clear();
    cout << "Optimization done." << endl;

    pcl::io::savePCDFile("result.pcd", *cloud);
    cout << "cloud saved result.pcd" << endl;

    getchar();

    return 0;
}
