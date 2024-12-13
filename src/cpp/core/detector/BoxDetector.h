#ifndef BOX_DETECTOR_H
#define BOX_DETECTOR_H

#include "core/box/Box.h"
#include "core/cluster/ClustersProcessor.h"
#include "utils/colors.h"
#include "utils/utils.h"

#include <opencv2/core.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class BoxDetector
{
public:
  BoxDetector() {};

  // Detect boxes
  cv::Mat detectBoxes(cv::Mat &edgesImage);

  // Filter and merge detected boxes
  cv::Mat filterBoxes(cv::Mat &detBoxesImage);

  // Infer missing boxes
  cv::Mat inferBoxes(cv::Mat &filtBoxesImage);

  // Generate segmentation masks
  cv::Mat segmentBoxes(cv::Mat &inputImage);

private:
  // Boxes detection threshold parameters
  const float contourAreaThreshold{200.0};
  const float maxAspectRatio{3.0f};
  const float minArea{8000.0f};
  const float maxArea{100000.0f};
  const float minPalletArea{300000.0f};

  std::vector<Box> detectedBoxes{};
  std::vector<Box> candidatePalletAreas{};

  // Average detected boxes properties
  float avgArea{0.0f};
  float avgAspectRatio{0.0f};
  const float histInterval{2.0f};
  float avgSemidiagonal{0.0f};
  int mostCommonAngle{0};

  Box palletBox{};

  // Pallet and box filtering standard deviations
  const float devPalletAngle{3.0f};
  const float devBoxAngle{4.0f};
  const float devAspectRatio{0.3f};
  const float devArea{0.4f};

  // Monte Carlo box inferring parameters
  const int nRandomPoints{20000};    // Monte Carlo number of random points
  const float expandFactorMin{1.1f}; // Factor expanding boxes for avoiding empty gaps
  const float expandFactorMax{1.3f}; // Factor expanding boxes for cluster number determination
  const int minPointsCluster{300};   // Minimum number of points for each cluster
  const float eps{1.3f};             // Factor multipliying avgSemidiagonal
  const size_t iterations{2};        // For reassigning points to closest clusters

  // Standard deviations for cluster box estimation
  const float devProjections{0.03f};
  const float devPerpDistances{0.01f};
  const float devAspectRatioCluster{0.36f};

  // Text properties for boxes numbering
  const int fontFace{cv::FONT_HERSHEY_DUPLEX};
  const double fontScale{1.4};
  const int thickness{2};
  int baseline{0};

  // Transparency factor for masks
  const double alpha{0.5};

  // Updates the average area, aspect ratio, angle, etc.
  void updateAvgBoxProperties();

  // Returns the points that are outside the expanded boxes
  std::vector<cv::Point2f> getPointsOutsideBoxes(const std::vector<cv::Point2f> &sampledPoints,
                                                 const std::vector<Box> &boxes,
                                                 const float expandFactor);

  // Returns the estimation of the number of clusters (one cluster per box)
  int getNumClusters(std::vector<cv::Point2f> sampledPoints);

  // Draws the boxes on an image
  void drawBoxes(cv::Mat &image, const std::vector<Box> &boxes, const cv::Scalar &color);
};

#endif // BOX_DETECTOR_H