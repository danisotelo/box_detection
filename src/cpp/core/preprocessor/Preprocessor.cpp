#include "core/preprocessor/Preprocessor.h"

/**
 * Read point cloud from binary file
 * @param input       : Input image data
 * @result pointCloud : Output extracted point cloud
 */
pcl::PointCloud<pcl::PointXYZRGB>::Ptr Preprocessor::extractPCL(const ImageData &input)
{
  // Extract the point cloud file names
  std::string pclFileName = input.imageName.substr(0, input.imageName.find_last_of('.')) + ".bin";
  std::string pclFilePath = dataFolder + '/' + pclFileName;

  // Check if the file exists
  if (!std::filesystem::exists(pclFilePath))
  {
    std::cerr << "Point cloud binary file does not exist: " << pclFilePath << '\n';
    return nullptr;
  }

  // Load the binary file
  std::ifstream inFile(pclFilePath, std::ios::binary);
  if (!inFile.is_open())
  {
    std::cerr << "Failed to open point cloud binary file: " << pclFilePath << '\n';
    return nullptr;
  }

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZRGB>());

  // Total number of points in the file
  inFile.seekg(0, std::ios::end);
  size_t fileSize = inFile.tellg();
  size_t totalPoints = fileSize / (sizeof(float) * 3 + sizeof(uint8_t) * 3);
  inFile.seekg(0, std::ios::beg);

  // Down-sample pointcloud
  double samplingRate{static_cast<double>(maxPoints) / totalPoints};
  std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  // Read points from binary file
  float x, y, z;
  uint8_t r, g, b;
  size_t sampledPoints = 0;

  while (inFile.read(reinterpret_cast<char *>(&x), sizeof(float)) &&
         inFile.read(reinterpret_cast<char *>(&y), sizeof(float)) &&
         inFile.read(reinterpret_cast<char *>(&z), sizeof(float)) &&
         inFile.read(reinterpret_cast<char *>(&r), sizeof(uint8_t)) &&
         inFile.read(reinterpret_cast<char *>(&g), sizeof(uint8_t)) &&
         inFile.read(reinterpret_cast<char *>(&b), sizeof(uint8_t)))
  {
    if (dist(rng) < samplingRate)
    {
      pcl::PointXYZRGB point;
      point.x = x;
      point.y = y;
      point.z = z;
      point.r = r;
      point.g = g;
      point.b = b;
      pointCloud->points.push_back(point);

      sampledPoints++;
      if (sampledPoints >= maxPoints)
      {
        break;
      }
    }
  }

  // Set cloud properties
  pointCloud->width = pointCloud->points.size();
  pointCloud->height = 1; // Unorganized cloud
  pointCloud->is_dense = true;

  return pointCloud;
}

/**
 * Voxel down-sample and SOR the point cloud
 * @param pointCloud : Input point cloud
 */
void Preprocessor::preprocessPCL(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pointCloud)
{
  // Voxel grid filter
  pcl::VoxelGrid<pcl::PointXYZRGB> voxelFilter;
  voxelFilter.setInputCloud(pointCloud);
  voxelFilter.setLeafSize(voxelSize, voxelSize, voxelSize);
  voxelFilter.filter(*pointCloud);

  // Statistical outlier removal
  statisticalOutlierRemoval(pointCloud, neighborsPre, stdRatioPre);
}

/**
 * RANSAC multiplane detection
 * @param pointCloud      : Input point cloud
 * @result detectedPlanes : RANSAC detected planes
 */
std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> Preprocessor::detectPlanes(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pointCloud)
{
  std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> detectedPlanes;
  for (int i = 0; i < nPlanes; ++i)
  {
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());

    pcl::SACSegmentation<pcl::PointXYZRGB> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(iterations);
    seg.setDistanceThreshold(planesThreshold);
    seg.setInputCloud(pointCloud);
    seg.segment(*inliers, *coefficients);

    // Extract plane
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr plane{new pcl::PointCloud<pcl::PointXYZRGB>()};
    for (const auto &index : inliers->indices)
    {
      plane->points.push_back(pointCloud->points[index]);
    }

    detectedPlanes.push_back(plane);

    // Remove plane points from pointCloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr newCloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    for (size_t j = 0; j < pointCloud->points.size(); ++j)
    {
      if (std::find(inliers->indices.begin(), inliers->indices.end(), j) == inliers->indices.end())
      {
        newCloud->points.push_back(pointCloud->points[j]);
      }
    }
    pointCloud = newCloud;
  }

  return detectedPlanes;
}

/**
 * Check if a point cloud plane has a pallet rectangular aspect ratio (~1.2)
 * @param plane  : Input point cloud
 * @result 0 / 1 : Boolean indicating if the aspect ratio is similar to the one of a pallet
 */
bool Preprocessor::hasRectangleAR(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &plane)
{
  // Compute the convex hull of the plane
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr hull(new pcl::PointCloud<pcl::PointXYZRGB>());
  pcl::ConvexHull<pcl::PointXYZRGB> convexHull;
  convexHull.setInputCloud(plane);
  convexHull.reconstruct(*hull);

  // Perform PCA on the convex hull
  Eigen::Vector4f centroid;
  pcl::compute3DCentroid(*hull, centroid);
  Eigen::Matrix3f covariance;
  pcl::computeCovarianceMatrixNormalized(*hull, centroid, covariance);

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigenSolver(covariance,
                                                             Eigen::ComputeEigenvectors);
  Eigen::Matrix3f eigenvectors = eigenSolver.eigenvectors();
  Eigen::Vector3f eigenvalues = eigenSolver.eigenvalues();

  // Project points onto principal axes and calculate dimensions
  float maxLength1 = 0.0f, maxLength2 = 0.0f;
  Eigen::Vector3f principalAxis1 = eigenvectors.col(2); // Largest eigenvector
  Eigen::Vector3f principalAxis2 = eigenvectors.col(1); // Second largest eigenvector

  for (const auto &point : hull->points)
  {
    Eigen::Vector3f p(point.x, point.y, point.z);
    float length1 = std::abs((p - centroid.head<3>()).dot(principalAxis1));
    float length2 = std::abs((p - centroid.head<3>()).dot(principalAxis2));
    maxLength1 = std::max(maxLength1, length1);
    maxLength2 = std::max(maxLength2, length2);
  }

  // Calculate aspect ratio
  float aspectRatio = std::max(maxLength1, maxLength2) /
                      std::min(maxLength1, maxLength2);

  return aspectRatio < aspectRatioThreshold;
}

/**
 * Calculates 2D projected mask from segmented 3D point cloud
 * @param plane : Input point cloud
 * @param input : Input image data
 * @result mask : Output 2D projected mask
 */
cv::Mat Preprocessor::calculateMask(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &plane,
                                    const cv::Size &imageSize)
{
  // Step 1: Project the 3D points of the plane to 2D
  std::vector<cv::Point2i> projectedPoints;
  for (const auto &point : plane->points)
  {
    Eigen::Vector3f point3D(point.x, point.y, point.z);
    Eigen::Vector3f projected = cam.getCameraMatrix() * point3D;

    // Convert to pixel coordinates
    int u = static_cast<int>(std::round(projected.x() / projected.z()));
    int v = static_cast<int>(std::round(projected.y() / projected.z()));

    // Ensure the pixel coordinates are within image bounds
    u = std::clamp(u, 0, imageSize.width - 1);
    v = std::clamp(v, 0, imageSize.height - 1);

    projectedPoints.emplace_back(u, v);
  }

  // Step 2: Compute the minimum enclosing rectangle
  cv::RotatedRect minRect = cv::minAreaRect(projectedPoints);

  // Step 3: Get the corner points of the rectangle and convert to int
  cv::Point2f rectPoints[4];
  minRect.points(rectPoints);
  std::vector<cv::Point> intRectPoints;
  for (int i = 0; i < 4; ++i)
  {
    intRectPoints.emplace_back(cv::Point(static_cast<int>(std::round(rectPoints[i].x)),
                                         static_cast<int>(std::round(rectPoints[i].y))));
  }

  // Step 4: Create a black mask and draw the rectangle on it in white
  cv::Mat mask = cv::Mat::zeros(imageSize, CV_8UC1);
  cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{intRectPoints},
               cv::Scalar(255));

  return mask;
}

/**
 * Masks the image (everything but the pallet)
 * @param input        : Input image
 * @result maskedImage : Masked image
 */
cv::Mat Preprocessor::maskImage(const ImageData &input)
{
  // Extract pointcloud
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud{Preprocessor::extractPCL(input)};

  // Voxel down-sample and remove outliers
  preprocessPCL(pointCloud);

  // Detect planes
  std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> detectedPlanes{detectPlanes(pointCloud)};

  // Generate mask for plane with rectangular aspect ratio
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr boxesPlane = nullptr;
  for (auto &plane : detectedPlanes)
  {
    statisticalOutlierRemoval(plane, neighborsPost, stdRatioPost);
    // If it has a rectangular aspect ratio (avoid ground plane and self in first image)
    if (hasRectangleAR(plane))
    {
      // Filter planes by average depth
      double avgDepth = 0.0;
      int pointCount = 0;

      for (const auto &point : plane->points)
      {
        avgDepth += point.z;
        ++pointCount;
      }

      avgDepth /= pointCount;
      if (avgDepth < maxDepth && avgDepth > minDepth)
      {
        boxesPlane = plane;
        break;
      }
    }
  }

  cv::Mat maskedImage{input.image.clone()};

  if (boxesPlane)
  {
    cv::Size imageShape{input.image.size()};

    // Apply mask to the original image
    cv::Mat mask{calculateMask(boxesPlane, imageShape)};
    cv::cvtColor(mask, mask, cv::COLOR_GRAY2BGR);

    cv::bitwise_and(input.image, mask, maskedImage);

    // View segmented cloud
    if (viewPCLFlag)
    {
      viewPCL(boxesPlane);
    }
  }
  else
  {
    std::cout << "No box plane detected. Retrying..." << std::endl;
    maskedImage = maskImage(input);
  }

  return maskedImage;
}

/**
 * Preprocesses the image (channel blur and edge detection)
 * @param input         : Input image
 * @return preprocImage : Preprocessed image
 */
cv::Mat Preprocessor::preprocessImage(const ImageData &input)
{
  cv::Mat preprocImage = input.image.clone();

  // Split input image into BGR channels
  std::vector<cv::Mat> channels;
  cv::split(preprocImage, channels);

  // Apply Gaussian blur and Canny edge detection to each channel
  for (const auto &channel : channels)
  {
    cv::GaussianBlur(channel, channel, cv::Size(3, 3), 0);
    cv::Canny(channel, channel, 40, 100);
  }

  // Combine edges from all channels
  preprocImage = channels[0] | channels[1] | channels[2];

  // Apply blur, dilation and erosion for better edge detection
  cv::GaussianBlur(preprocImage, preprocImage, cv::Size(3, 3), 1);
  cv::dilate(preprocImage, preprocImage, cv::Mat(), cv::Point(-1, -1), 1);
  cv::erode(preprocImage, preprocImage, cv::Mat(), cv::Point(-1, -1), 1);

  return preprocImage;
}