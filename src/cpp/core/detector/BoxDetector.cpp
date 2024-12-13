#include "core/detector/BoxDetector.h"

/**
 * Detects boxes from edges image
 * @param edgesImage     : Edges image
 * @return detBoxesImage : Output detected boxes image
 */
cv::Mat BoxDetector::detectBoxes(cv::Mat &edgesImage)
{
  // Find contours in the edges image
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(edgesImage, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

  // Vector to store detected boxes
  std::vector<Box> boxes;
  std::vector<Box> maybePallets;

  // Approximate each contour with a rotated rectangle
  for (const auto &contour : contours)
  {
    // Skip too small contours
    float contourArea = cv::contourArea(contour);
    if (contourArea < contourAreaThreshold)
    {
      continue;
    }

    // Create a box and filter by area and aspect ratio
    Box box{contour};

    if (box.aspectRatio < maxAspectRatio && box.area > minArea && box.area < maxArea)
    {
      // Check if the current box is inside an existing box
      bool isInsideAnotherBox = false;
      for (const auto &existingBox : boxes)
      {
        if (box.isInsideOf(existingBox))
        {
          isInsideAnotherBox = true;
          break;
        }
      }

      // Add the box if it's not inside another box
      if (!isInsideAnotherBox)
      {
        // Remove smaller boxes inside the new box
        boxes.erase(std::remove_if(boxes.begin(), boxes.end(),
                                   [&box](const Box &existingBox)
                                   {
                                     return existingBox.isInsideOf(box);
                                   }),
                    boxes.end());

        boxes.push_back(box);
      }
    }
    // Filter candidates to be the pallet box
    else if (box.area > minPalletArea)
    {
      maybePallets.push_back(box);
    }
  }

  detectedBoxes = boxes;
  candidatePalletAreas = maybePallets;

  // Update parameters and draw boxess
  updateAvgBoxProperties();
  cv::Mat detBoxesImage{edgesImage.clone()};
  cv::cvtColor(detBoxesImage, detBoxesImage, cv::COLOR_GRAY2BGR);
  drawBoxes(detBoxesImage, detectedBoxes, Colors::yellow);

  return detBoxesImage;
};

/**
 * Filter detected boxes
 * @param detBoxesImage   : Detected boxes input image
 * @return filtBoxesImage : Output filtered boxes image
 */
cv::Mat BoxDetector::filterBoxes(cv::Mat &detBoxesImage)
{
  // Filter candidate pallet boxes by angle to find the pallet box
  for (const auto &pallet : candidatePalletAreas)
  {
    if (pallet.isAngleValid(mostCommonAngle, devPalletAngle))
    {
      palletBox = pallet;
      break;
    }
  }

  // Filter boxes by proximity to average aspect ratio and angle
  std::vector<Box> boxes;
  std::vector<Box> maybeFlaps;
  for (const auto &box : detectedBoxes)
  {
    if (box.isAngleValid(mostCommonAngle, devBoxAngle))
    {
      if (box.isAspectRatioValid(avgAspectRatio, devAspectRatio))
      {
        boxes.push_back(box);
      }
      else
      {
        maybeFlaps.push_back(box); // To account for box flaps
      }
    }
  }

  // Try merging two flaps to check if they form a box
  for (const auto &flap1 : maybeFlaps)
  {
    for (const auto &flap2 : maybeFlaps)
    {
      Box mergedBox = flap1.mergeWith(flap2);
      if (mergedBox.isAspectRatioValid(avgAspectRatio, devAspectRatio) &&
          mergedBox.isAreaValid(avgArea, devArea))
      {
        boxes.push_back(mergedBox);
      }
    }
  }

  detectedBoxes = boxes;

  // Update properties and draw boxes
  updateAvgBoxProperties();
  cv::Mat filtBoxesImage{detBoxesImage.clone()};
  drawBoxes(filtBoxesImage, detectedBoxes, Colors::magenta);
  palletBox.draw(filtBoxesImage, Colors::blue);

  return filtBoxesImage;
};

/**
 * Infer boxes from detected ones using Monte Carlo method and clustering
 * @param  filtBoxesImage  : Image with filtered detected boxes
 * @return inferBoxesImage : Modified image with added inferred boxes and clusters
 */
cv::Mat BoxDetector::inferBoxes(cv::Mat &filtBoxesImage)
{
  // Generate n random points inside the pallet area
  std::vector<cv::Point2f> sampledPoints{palletBox.getRandomPointsInside(nRandomPoints)};

  // Get sampled points outside already detected boxes
  std::vector<cv::Point2f> pointsOutsideBoxes{getPointsOutsideBoxes(sampledPoints,
                                                                    detectedBoxes,
                                                                    expandFactorMin)};
  // Estimation of the number clusters (boxes to infer)
  int maxClusters{getNumClusters(sampledPoints)};

  cv::Mat inferBoxesImage{filtBoxesImage.clone()};
  if (maxClusters != 0)
  {
    // Create a ClustersProcessor object and apply density-based clustering (DBSCAN)
    const float distanceThreshold{eps * avgSemidiagonal};
    ClustersProcessor clusterProcessor{distanceThreshold, minPointsCluster};
    clusterProcessor.performClustering(pointsOutsideBoxes);

    // Reduce clusters to maxClusters
    clusterProcessor.reduceClusters(maxClusters);

    // Reassign points to their closest clusters iteratively
    clusterProcessor.reassignPointsToClosestClusters(iterations);

    for (const auto &cluster : clusterProcessor.getClusters())
    {
      // Get closest box line to cluster
      auto [closestStart, closestEnd] = cluster.findClosestLine(detectedBoxes);
      cv::Point2f lineVector = closestEnd - closestStart;

      // Compute projections and perpendicular distances
      auto [projections, perpDistances] = cluster.computeProjAndDist(closestStart, closestEnd);

      // Filter projections and compute box dimensions
      cv::Size2f boxSize = cluster.getBoxDim(projections, perpDistances,
                                             devProjections, devPerpDistances);

      // Create cluster box
      float angleBox{static_cast<float>(std::atan2(lineVector.y, lineVector.x) * 180.0 / CV_PI)};
      Box clusterBox{cluster.getCentroid(), boxSize, angleBox};

      // Add to detected box if it passes aspect ratio and area filter
      if (clusterBox.isAspectRatioValid(avgAspectRatio, devAspectRatioCluster) &&
          clusterBox.isAreaValid(avgArea, devArea))
      {
        detectedBoxes.push_back(clusterBox);
      }

      // Draw cluster and box
      cv::Scalar color{Colors::randomColor()};
      cluster.draw(inferBoxesImage, color);
      clusterBox.draw(inferBoxesImage, color);
      cv::line(inferBoxesImage, closestEnd, closestStart, Colors::white, 6);
    }
  }

  return inferBoxesImage;
}

/**
 * Apply segmentation masks to the image
 * @param inputImage  : Input image over which the mask is applied
 * @return inputImage : Modified input image
 */
cv::Mat BoxDetector::segmentBoxes(cv::Mat &inputImage)
{
  // Clone the input image to create an overlay
  cv::Mat overlay{inputImage.clone()};

  // Draw each box with a different color
  size_t boxNumber = 1;
  for (const auto &box : detectedBoxes)
  {
    // Create a mask by filling the box with the color
    cv::Scalar color{Colors::randomColor()};
    cv::fillPoly(overlay, box.contours, color);

    // Draw the border of the box with the same color
    box.draw(inputImage, color);

    // Add text in the center of the box
    cv::Size textSize = cv::getTextSize(std::to_string(boxNumber), fontFace,
                                        fontScale, thickness, &baseline);
    cv::Point2f textPosition(box.center.x - textSize.width / 2.0,
                             box.center.y + textSize.height / 2.0);
    cv::putText(inputImage, std::to_string(boxNumber), textPosition, fontFace,
                fontScale, Colors::makeDarker(color), thickness);

    boxNumber++;
  }

  // Blend the overlay with the original image
  cv::addWeighted(overlay, alpha, inputImage, 1.0 - alpha, 0, inputImage);

  return inputImage;
};

/**
 * Updates the average area, aspect ratio, angle, and semidiagonal
 * of the detected boxes.
 */
void BoxDetector::updateAvgBoxProperties()
{
  size_t nBoxes = detectedBoxes.size();
  avgArea = 0.0f;
  avgAspectRatio = 0.0f;
  std::map<int, int> angleHistogram;

  // Calculate average area and aspect ratio
  for (const auto &box : detectedBoxes)
  {
    avgArea += box.area;
    avgAspectRatio += box.aspectRatio;
    float angle{box.angle};
    int roundedAngle{static_cast<int>(std::round(angle / histInterval) * histInterval)};
    angleHistogram[roundedAngle]++;
  }

  if (nBoxes != 0)
  {
    avgArea /= nBoxes;
    avgAspectRatio /= nBoxes;
  }

  // Update the box average semidiagonal (useful for clustering)
  avgSemidiagonal = avgAspectRatio / 2 * sqrt(avgArea / avgAspectRatio);

  // Find the most common angle
  mostCommonAngle = std::max_element(angleHistogram.begin(), angleHistogram.end(),
                                     [](const std::pair<int, int> &a,
                                        const std::pair<int, int> &b)
                                     { return a.second < b.second; })
                        ->first;
}

/**
 * Returns the points that are outside the expanded boxes
 * @param sampledPoints       : Input points
 * @param boxes               : Input boxes
 * @param expandFactor        : Desired boxes expansion factor
 * @return pointsOutsideBoxes : Points outside the expanded boxes
 */
std::vector<cv::Point2f> BoxDetector::getPointsOutsideBoxes(const std::vector<cv::Point2f> &sampledPoints,
                                                            const std::vector<Box> &boxes,
                                                            const float expandFactor)
{
  std::vector<cv::Point2f> pointsOutsideBoxes;
  for (const auto &point : sampledPoints)
  {
    bool isOutsideAllBoxes{true};
    for (const auto &box : boxes)
    {
      // Expand the boxes to avoid gaps between them
      Box expandedBox{box.expand(expandFactor)};
      if (cv::pointPolygonTest(expandedBox.vertices, point, false) >= 0)
      {
        isOutsideAllBoxes = false;
        break;
      }
    }
    if (isOutsideAllBoxes)
    {
      pointsOutsideBoxes.push_back(point);
    }
  }

  return pointsOutsideBoxes;
}

/**
 * Returns the estimation of the number of clusters (one cluster per box)
 * @param sampledPoints : Input pointsç
 * @return maxClusters  : Number of clusters estimation
 */
int BoxDetector::getNumClusters(std::vector<cv::Point2f> sampledPoints)
{
  // Get sampled points outside already detected boxes
  std::vector<cv::Point2f> pointsOutsideBoxes{getPointsOutsideBoxes(sampledPoints,
                                                                    detectedBoxes,
                                                                    expandFactorMax)};
  // Round up since the boxes are being expanded
  double pointsPerBox{nRandomPoints * avgArea / (palletBox.width * palletBox.height)};
  int maxClusters{0};
  maxClusters = ceil(pointsOutsideBoxes.size() / pointsPerBox);

  return maxClusters;
}

/**
 * Draws the boxes on an image
 * @param image : Image where it is desired to draw the box
 * @param boxes : Boxes to draw
 * @param color : Box color
 */
void BoxDetector::drawBoxes(cv::Mat &image, const std::vector<Box> &boxes, const cv::Scalar &color)
{
  for (const auto &box : boxes)
  {
    box.draw(image, color);
  }
}