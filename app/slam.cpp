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
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/slam3d/edge_se3.h>

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/visualization/cloud_viewer.h>

#include "slam_base.h"
#include "tum_data_rgbd.h"

// 估计一个运动的大小
double normofTransform(cv::Mat rvec, cv::Mat tvec);

// 检测两个帧，结果定义
enum CHECK_RESULT { NOT_MATCHED = 0,
                    TOO_FAR_AWAY,
                    TOO_CLOSE,
                    KEYFRAME };
// 函数声明
CHECK_RESULT checkKeyframes(FRAME& f1, FRAME& f2, g2o::SparseOptimizer& opti, bool is_loops = false);
// 检测近距离的回环
void checkNearbyLoops(vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti);
// 随机检测回环
void checkRandomLoops(vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti);

CAMERA_INTRINSIC_PARAMETERS camera;

int main(int argc, char** argv) {
    ParameterReader pd;
    int idx_s = atoi(pd.getData("start_index").c_str());
    int idx_e = atoi(pd.getData("end_index").c_str());
    bool visualize = pd.getData("visualize_pointcloud") == string("yes");
    int min_inliers = atoi(pd.getData("min_inliers").c_str());
    double max_norm = atof(pd.getData("max_norm").c_str());
    double keyframe_threshold = atof(pd.getData("keyframe_threshold").c_str());
    bool check_loop_closure = pd.getData("check_loop_closure") == string("yes");
    double gridsize = atof(pd.getData("voxel_grid").c_str());  //分辨图可以在parameters.txt里调

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

    int idx_c;
    FRAME frame_c;

    vector<FRAME> keyframes;

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

            frame_c = FRAME(img_color, img_depth, idx_c);
            compute_keypoints_desp(frame_c);

            // 向globalOptimizer增加第一个顶点
            g2o::VertexSE3* v = new g2o::VertexSE3();
            v->setId(idx_c);
            v->setEstimate(Eigen::Isometry3d::Identity());  //估计为单位矩阵
            v->setFixed(true);                              //第一个顶点固定，不用优化
            globalOptimizer.addVertex(v);

            keyframes.push_back(frame_c);

            continue;
        }

        cout << "current index: " << idx_c << endl;
        frame_c = FRAME(img_color, img_depth, idx_c);
        compute_keypoints_desp(frame_c);

        CHECK_RESULT result = checkKeyframes(keyframes.back(), frame_c, globalOptimizer);  //匹配该帧与keyframes里最后一帧
        switch (result)                                                                    // 根据匹配结果不同采取不同策略
        {
            case NOT_MATCHED:
                //没匹配上，直接跳过
                cout << "Not enough inliers." << endl;
                break;
            case TOO_FAR_AWAY:
                // 太近了，也直接跳
                cout << "Too far away, may be an error." << endl;
                break;
            case TOO_CLOSE:
                // 太远了，可能出错了
                cout << "Too close, not a keyframe" << endl;
                break;
            case KEYFRAME:
                cout << "This is a new keyframe" << endl;
                // 不远不近，刚好
                /**
             * This is important!!
             * This is important!!
             * This is important!!
             * (very important so I've said three times!)
             */
                // 检测回环
                if (check_loop_closure) {
                    checkNearbyLoops(keyframes, frame_c, globalOptimizer);
                    checkRandomLoops(keyframes, frame_c, globalOptimizer);
                }
                keyframes.push_back(frame_c);
                break;
            default:
                break;
        }

        usleep(30000);
    }

    // 优化所有边
    cout << "optimizing pose graph, vertices: " << globalOptimizer.vertices().size() << endl;
    globalOptimizer.save("result_before.g2o");
    globalOptimizer.initializeOptimization();
    globalOptimizer.optimize(100);  //可以指定优化步数
    globalOptimizer.save("result_after.g2o");
    cout << "Optimization done." << endl;

    // 拼接点云地图
    cout << "saving the point cloud map..." << endl;
    PointCloud::Ptr output(new PointCloud());  //全局地图
    PointCloud::Ptr tmp(new PointCloud());

    pcl::PassThrough<PointT> pass;
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0.0, 4.0);  // 4m以上就不要了

    pcl::VoxelGrid<PointT> voxel;  // 网格滤波器，调整地图分辨率
    voxel.setLeafSize(gridsize, gridsize, gridsize);

    for (size_t i = 0; i < keyframes.size(); i++) {
        // 从g2o里取出一帧
        g2o::VertexSE3* vertex = dynamic_cast<g2o::VertexSE3*>(globalOptimizer.vertex(keyframes[i].frameID));
        Eigen::Isometry3d pose = vertex->estimate();  //该帧优化后的位姿
        PointCloud::Ptr newCloud = image_to_pointcloud(keyframes[i].rgb, keyframes[i].depth, camera);
        voxel.setInputCloud(newCloud);
        voxel.filter(*tmp);
        pass.setInputCloud(tmp);
        pass.filter(*newCloud);

        pcl::transformPointCloud(*newCloud, *tmp, pose.matrix());  // 把点云变换后加入全局地图中
        *output += *tmp;

        tmp->clear();
        newCloud->clear();
    }

    voxel.setInputCloud(output);
    voxel.filter(*tmp);

    pcl::io::savePCDFile("result.pcd", *tmp);
    cout << "Final map is saved result.pcd" << endl;

    globalOptimizer.clear();

    return 0;
}

double normofTransform(cv::Mat rvec, cv::Mat tvec) {
    return fabs(min(cv::norm(rvec), 2 * M_PI - cv::norm(rvec))) + fabs(cv::norm(tvec));
}

CHECK_RESULT checkKeyframes(FRAME& f1, FRAME& f2, g2o::SparseOptimizer& opti, bool is_loops) {
    static ParameterReader pd;
    static int min_inliers = atoi(pd.getData("min_inliers").c_str());
    static double max_norm = atof(pd.getData("max_norm").c_str());
    static double keyframe_threshold = atof(pd.getData("keyframe_threshold").c_str());
    static double max_norm_lp = atof(pd.getData("max_norm_lp").c_str());
    static g2o::RobustKernel* robustKernel = g2o::RobustKernelFactory::instance()->construct("Cauchy");
    // 比较f1 和 f2
    RESULT_OF_PNP result = estimate_motion(f1, f2, camera);
    if (result.inliers < min_inliers)  //inliers不够，放弃该帧
        return NOT_MATCHED;
    // 计算运动范围是否太大
    double norm = normofTransform(result.rvec, result.tvec);
    if (is_loops == false) {
        if (norm >= max_norm)
            return TOO_FAR_AWAY;  // too far away, may be error
    } else {
        if (norm >= max_norm_lp)
            return TOO_FAR_AWAY;
    }

    if (norm <= keyframe_threshold) return TOO_CLOSE;  // too adjacent frame

    // 向g2o中增加这个顶点与上一帧联系的边
    // 顶点部分
    // 顶点只需设定id即可
    if (is_loops == false) {
        g2o::VertexSE3* v = new g2o::VertexSE3();
        v->setId(f2.frameID);
        v->setEstimate(Eigen::Isometry3d::Identity());
        opti.addVertex(v);
    }

    g2o::EdgeSE3* edge = new g2o::EdgeSE3();
    edge->vertices()[0] = opti.vertex(f1.frameID);
    edge->vertices()[1] = opti.vertex(f2.frameID);
    edge->setRobustKernel(robustKernel);
    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
    // 信息矩阵是协方差矩阵的逆，表示我们对边的精度的预先估计
    // 因为pose为6D的，信息矩阵是6*6的阵，假设位置和角度的估计精度均为0.1且互相独立
    // 那么协方差则为对角为0.01的矩阵，信息阵则为100的矩阵
    // 也可以将角度设大一些，表示对角度的估计更加准确
    information(0, 0) = information(1, 1) = information(2, 2) = 100;
    information(3, 3) = information(4, 4) = information(5, 5) = 100;
    edge->setInformation(information);
    Eigen::Isometry3d T = cvmat_to_eigen(result.rvec, result.tvec);
    edge->setMeasurement(T.inverse()); // 边的估计即是pnp求解之结果
    opti.addEdge(edge);

    return KEYFRAME;
}

void checkNearbyLoops(vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti) {
    static ParameterReader pd;
    static int nearby_loops = atoi(pd.getData("nearby_loops").c_str());

    // 就是把currFrame和 frames里末尾几个测一遍
    if (frames.size() <= nearby_loops) {
        // no enough keyframes, check everyone
        for (size_t i = 0; i < frames.size(); i++) {
            checkKeyframes(frames[i], currFrame, opti, true);
        }
    } else {
        // check the nearest ones
        for (size_t i = frames.size() - nearby_loops; i < frames.size(); i++) {
            checkKeyframes(frames[i], currFrame, opti, true);
        }
    }
}

void checkRandomLoops(vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti) {
    static ParameterReader pd;
    static int random_loops = atoi(pd.getData("random_loops").c_str());
    srand((unsigned int)time(NULL));
    // 随机取一些帧进行检测

    if (frames.size() <= random_loops) {
        // no enough keyframes, check everyone
        for (size_t i = 0; i < frames.size(); i++) {
            checkKeyframes(frames[i], currFrame, opti, true);
        }
    } else {
        // randomly check loops
        for (int i = 0; i < random_loops; i++) {
            int index = rand() % frames.size();
            checkKeyframes(frames[index], currFrame, opti, true);
        }
    }
}
