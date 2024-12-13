#include "core/BoxSegmenter.h"
#include <iostream>
#include <string>

int main()
{
  // Define the data and input folders
  std::string dataFolder = "../Data";
  std::string outputFolder = "../Results";

  bool saveIntResults = true; // Flag to save intermediate results
  bool viewPCL = true;        // Flag to visualize segmented point clouds

  // Create BoxSegmenter object
  BoxSegmenter boxSegmenter(dataFolder, outputFolder, saveIntResults, viewPCL);

  // Process a single image or all images
  std::string processType;
  std::cout << "Enter process type ('single' / 'all'): ";
  std::cin >> processType;

  while (true)
  {
    if (processType == "single")
    {
      std::string imageName;
      std::cout << "Enter the image file name (e.g., 1.png): ";
      std::cin >> imageName;

      if (boxSegmenter.processImage(imageName))
      {
        break; // Valid input, exit the loop
      }
      else
      {
        std::cout << "Invalid image file name. Please try again.\n";
      }
    }
    else if (processType == "all")
    {
      boxSegmenter.processAllImages();
      break;
    }
    else
    {
      std::cout << "Invalid process type. Please enter 'single' or 'all': ";
      std::cin >> processType;
    }
  }

  return 0;
}