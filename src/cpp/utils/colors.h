#ifndef COLORS_H
#define COLORS_H

#include <opencv2/opencv.hpp>

namespace Colors
{
    const cv::Scalar yellow = cv::Scalar(58, 187, 242);
    const cv::Scalar black = cv::Scalar(0, 0, 0);
    const cv::Scalar white = cv::Scalar(255, 255, 255);
    const cv::Scalar blue = cv::Scalar(252, 186, 3);
    const cv::Scalar magenta = cv::Scalar(110, 61, 245);

    // Function to generate a random color
    inline cv::Scalar randomColor()
    {
        return cv::Scalar(cv::theRNG().uniform(50, 255),
                          cv::theRNG().uniform(50, 255),
                          cv::theRNG().uniform(50, 255));
    }

    // Function to return a darker version of a color
    inline cv::Scalar makeDarker(const cv::Scalar &color, double factor = 0.3)
    {
        // Ensure the factor is within valid bounds
        factor = std::clamp(factor, 0.0, 1.0);
        return cv::Scalar(color[0] * factor, color[1] * factor, color[2] * factor);
    }
}

#endif // COLORS_H