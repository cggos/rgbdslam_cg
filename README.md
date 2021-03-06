# rgbdslam_cg

根据高翔博客 [一起做RGB-D SLAM](https://www.cnblogs.com/gaoxiang12/tag/%E4%B8%80%E8%B5%B7%E5%81%9ARGB-D%20SLAM/)（其代码在 [gaoxiang12/rgbd-slam-tutorial-gx](https://github.com/gaoxiang12/rgbd-slam-tutorial-gx)），从零搭建 RGBD-SLAM。

-----

# Dependencies

* Eigen3
* OpenCV3
* PCL 1.7
* G2O (the lastest version)

# Build

```sh
mkdir build & cd build
cmake ..
make -j2
```

# Run

```sh
cd build

../bin/generate_pointcloud

../bin/detect_features

../bin/join_pointcloud

../bin/visual_odometry

../bin/rgbdslam
```

# Result

* result of `bin/visual_odometry`
  <div align="center">
    <img src="images/run_rgbdslam.jpg"/>
  </div>

* `g2o_viewer result_after.g2o`, the result of `bin/rgbdslam`
  <div align="center">
    <img src="images/g2oviewer_rgbdslam.jpg"/>
  </div>