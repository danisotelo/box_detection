#ifndef UTILS_H
#define UTILS_H

#include "colors.h"

#include <Eigen/Dense>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/visualization/cloud_viewer.h>

struct ImageData
{
    const std::string &folderName;
    const std::string &imageName;
    const cv::Mat &image;
};

// Creates a folder if it doesn't exist
void createFolder(const std::string &folderPath);

// Saves an image
void saveImage(const std::string &outputPath, const ImageData &imageData);

// Apply Statistical Outlier Removal (SOR) to a pointcloud
void statisticalOutlierRemoval(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pointCloud,
                               int neighbors,
                               double stdRatio);

// Point cloud visualizer
void viewPCL(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pointCloud);

// Camera object
class Camera
{
public:
    Camera(float f_x, float f_y, float c_x, float c_y)
        : f_x(f_x), f_y(f_y), c_x(c_x), c_y(c_y),
          cameraMatrix((Eigen::Matrix3f() << f_x, 0.0, c_x, 0.0, f_y, c_y, 0.0, 0.0, 1.0).finished()) {}

    Eigen::Matrix3f getCameraMatrix() const { return cameraMatrix; }

private:
    const float f_x;
    const float f_y;
    const float c_x;
    const float c_y;

    const Eigen::Matrix3f cameraMatrix;
};

#endif // UTILS_H