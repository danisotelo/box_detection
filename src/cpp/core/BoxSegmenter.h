#ifndef BOX_SEGMENTER_H
#define BOX_SEGMENTER_H

#include "core/detector/BoxDetector.h"
#include "core/preprocessor/Preprocessor.h"
#include "utils/utils.h"

#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class BoxSegmenter
{
private:
  std::string dataFolder;
  std::string outputFolder;

  const bool saveInt; // Save intermediate results flag
  const bool viewPCL; // View filtered point cloud

  Preprocessor preprocessor; // Preprocessor object to mask and detect edges
  BoxDetector boxDetector;   // Box detector object to detect and segment boxes

public:
  BoxSegmenter(const std::string &dataFolder, const std::string &outputFolder,
               const bool saveInt, const bool viewPCL);

  // Function to process a single image
  bool processImage(const std::string &imageName);

  // Function to process all images in the data folder
  void processAllImages();
};

#endif // BOX_SEGMENTER_H