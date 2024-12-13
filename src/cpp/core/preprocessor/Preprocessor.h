#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#include "utils/utils.h"

#include <pcl/common/centroid.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/surface/convex_hull.h>
#include <random>

class Preprocessor
{
public:
  Preprocessor(const std::string &dataFolder, const bool viewPCLFlag)
      : dataFolder(dataFolder), viewPCLFlag(viewPCLFlag) {};

  // Function to mask the image
  cv::Mat maskImage(const ImageData &input);

  // Function to preprocess an image (grayscale, blur, and edge detection)
  cv::Mat preprocessImage(const ImageData &input);

private:
  const std::string dataFolder;
  const bool viewPCLFlag;

  // Intrinsic camera parameters
  const float f_x{975.482117};
  const float f_y{975.301147};
  const float c_x{1019.53790};
  const float c_y{776.480408};
  Camera cam{f_x, f_y, c_x, c_y};

  // Maximum point cloud points (down-sample)
  const int maxPoints{80000};

  // Voxel down-sample and outlier removal parameters
  const float voxelSize{4.0f};
  const int neighborsPre{50};
  const double stdRatioPre{0.5};

  // Plane detection parameters
  const int nPlanes{3};
  const double planesThreshold{8.0};
  const int iterations{500};

  // Rectangular plane aspect ratio threshold
  const float aspectRatioThreshold{1.3f};

  // Second outlier removal parameters
  const int neighborsPost{130};
  const double stdRatioPost{0.3};

  // Plane average depth filtering parameters
  const double minDepth{1100.0};
  const double maxDepth{2400.0};

  // Extract point cloud from binary file
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr extractPCL(const ImageData &input);

  // Voxel down-sample and remove outliers
  void preprocessPCL(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pointCloud);

  // Detect planes in point cloud using RANSAC
  std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> detectPlanes(
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pointCloud);

  // Detect if the filtered point cloud has a pallet aspect ratio
  bool hasRectangleAR(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &plane);

  // Calculates the binary mask corresponding to the pallet pointcloud
  cv::Mat calculateMask(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &plane,
                        const cv::Size &imageSize);
};

#endif // PREPROCESSOR_H