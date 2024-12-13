#include "core/BoxSegmenter.h"

// Constructor
BoxSegmenter::BoxSegmenter(const std::string &dataFolder,
                           const std::string &outputFolder,
                           const bool saveInt, const bool viewPCL)
    : dataFolder(dataFolder), outputFolder(outputFolder),
      saveInt(saveInt), viewPCL(viewPCL),
      preprocessor(dataFolder, viewPCL) {}

/**
 * Process a single image
 * @param imageName : Input image name
 * @return 0 or 1   : Returns 0 if the image is not found
 */
bool BoxSegmenter::processImage(const std::string &imageName)
{
  createFolder(outputFolder);

  std::string inputPath = dataFolder + '/' + imageName;
  cv::Mat inputImage = cv::imread(inputPath);
  if (inputImage.empty())
  {
    std::cerr << "Error loading image: " << inputPath << '\n';
    return 0;
  }

  ImageData inputImageData{"input", imageName, inputImage};
  std::vector<ImageData> intermediateImgs{};

  // Mask the image
  cv::Mat maskedImage{preprocessor.maskImage(inputImageData)};
  ImageData maskedImageData{"masked", imageName, maskedImage};
  intermediateImgs.push_back(maskedImageData);

  // Preprocess the image
  cv::Mat edgesImage{preprocessor.preprocessImage(maskedImageData)};
  ImageData edgesImageData{"edges", imageName, edgesImage};
  intermediateImgs.push_back(edgesImageData);

  // Detect boxes
  cv::Mat detBoxesImage{boxDetector.detectBoxes(edgesImage)};
  ImageData detBoxesImageData{"det_boxes", imageName, detBoxesImage};
  intermediateImgs.push_back(detBoxesImageData);

  // Filtered boxes
  cv::Mat filtBoxesImage{boxDetector.filterBoxes(detBoxesImage)};
  ImageData filtBoxesImageData{"filtered_boxes", imageName, filtBoxesImage};
  intermediateImgs.push_back(filtBoxesImageData);

  // Inferred boxes
  cv::Mat inferBoxesImage{boxDetector.inferBoxes(filtBoxesImage)};
  ImageData inferBoxesImageData{"inferred_boxes", imageName, inferBoxesImage};
  intermediateImgs.push_back(inferBoxesImageData);

  // Save intermediate results
  if (saveInt)
  {
    for (const auto &imageData : intermediateImgs)
    {
      saveImage(outputFolder, imageData);
    }
  }

  // Segmented boxes
  cv::Mat segmentBoxesImage{boxDetector.segmentBoxes(inputImage)};
  ImageData segmentBoxesImageData{"output", imageName, segmentBoxesImage};
  saveImage(outputFolder, segmentBoxesImageData);

  return 1;
}

/**
 * Process all the images of the folder
 */
void BoxSegmenter::processAllImages()
{
  createFolder(outputFolder);
  for (const auto &entry : std::filesystem::directory_iterator(dataFolder))
  {
    if (entry.is_regular_file() && entry.path().extension() == ".png")
    {
      std::string imageName = entry.path().filename().string();
      std::cout << "Processing " << imageName << " ...\n";
      processImage(imageName);
    }
  }
}