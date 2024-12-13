#ifndef BOX_H
#define BOX_H

#include <opencv2/opencv.hpp>

class Box
{
public:
    float width{0.0f};
    float height{0.0f};
    float area{0.0f};
    float aspectRatio{0.0f};
    float angle{0.0f};
    cv::Point2f center{};
    std::vector<cv::Point2f> vertices{4};
    std::vector<std::vector<cv::Point>> contours{};

    // Constructors
    Box() {};
    Box(const std::vector<cv::Point> &contour);
    Box(const std::vector<cv::Point2f> &contour);
    Box(const cv::Point2f &centroid, const cv::Size2f &dimensions, const float angle);

    // Checks if the box is inside another one
    bool isInsideOf(const Box &other) const;

    // Merges the box with another one
    Box mergeWith(const Box &other) const;

    // Expands the box with a factor
    Box expand(const float factor) const;

    // Draws box on an image
    void draw(cv::Mat &image, const cv::Scalar &color) const;

    // Generates random points inside the box
    std::vector<cv::Point2f> getRandomPointsInside(const int n);

    // Checks if an angle is close enough to another or its perpendiculars
    bool isAngleValid(const int avgAngle, const float devAngle) const;

    // Checks if the area is close enough to another value
    bool isAreaValid(const float avgArea, const float devArea) const;

    // Checks if the aspect ratio is close enough to another value
    bool isAspectRatioValid(const float avgAspectRatio, const float devAspectRatio) const;

private:
    cv::RotatedRect rectangle;

    // Helper constructor function
    void initializeContours();
};

#endif // BOX_H
