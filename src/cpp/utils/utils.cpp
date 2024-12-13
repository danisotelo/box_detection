#include "utils.h"

/**
 * Creates a folder if it does not exist
 * @param folderPath: Path to folder to create
 */
void createFolder(const std::string &folderPath)
{
  if (!std::filesystem::exists(folderPath))
  {
    std::filesystem::create_directories(folderPath);
  }
}

/**
 * Saves image
 * @param outputPath : Path to output folder
 * @param folderName : Name of the folder where to save
 * @param imageName  : Name of the image to save
 * @param image      : Image to save
 */
void saveImage(const std::string &outputPath, const ImageData &imageData)
{
  std::string folderPath = outputPath + '/' + imageData.folderName;
  createFolder(folderPath);

  // Construct new file name
  std::string baseName = imageData.imageName.substr(0, imageData.imageName.find_last_of('.'));
  std::string outputFile =
      folderPath + '/' + baseName + '_' + imageData.folderName + ".png";
  cv::imwrite(outputFile, imageData.image);
}

/**
 * Applies Statistical Outlier Removal (SOR) to a pointcloud
 * @param pointCloud : Pointcloud to which the SOR is applied
 * @param neighbors  : The number of points to use for mean distance estimation
 * @param stdRatio   : Standard deviation multiplier for distance threshold calculation
 */
void statisticalOutlierRemoval(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pointCloud,
                               int neighbors,
                               double stdRatio)
{
  pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
  sor.setInputCloud(pointCloud);
  sor.setMeanK(neighbors);
  sor.setStddevMulThresh(stdRatio);
  sor.filter(*pointCloud);
}

/**
 * Point cloud visualizer
 * @param pointCloud : Point cloud to visualize
 */
void viewPCL(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pointCloud)
{
  // Create a PCLVisualizer object
  pcl::visualization::PCLVisualizer::Ptr viewer(
      new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));

  // Set the background color
  viewer->setBackgroundColor(0, 0, 0);

  // Add the point cloud to the viewer
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(pointCloud);
  viewer->addPointCloud<pcl::PointXYZRGB>(pointCloud, rgb, "Point Cloud Viewer");

  viewer->spin();
}
