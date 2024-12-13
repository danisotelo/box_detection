#include "core/box/Box.h"

Box::Box(const std::vector<cv::Point> &contour)
    : Box(std::vector<cv::Point2f>(contour.begin(), contour.end())) {}

Box::Box(const std::vector<cv::Point2f> &contour)
    : rectangle(cv::minAreaRect(contour))
{
    width = rectangle.size.width;
    height = rectangle.size.height;
    area = rectangle.size.area();
    aspectRatio = std::max(width, height) / std::min(width, height);
    angle = rectangle.angle;
    center = rectangle.center;
    vertices.resize(4);
    rectangle.points(vertices.data());
    initializeContours();
}

Box::Box(const cv::Point2f &centroid, const cv::Size2f &dimensions, float angle)
    : rectangle(cv::RotatedRect(centroid, dimensions, angle))
{
    width = rectangle.size.width;
    height = rectangle.size.height;
    area = abs(width * height);
    aspectRatio = abs(std::max(width, height) / std::min(width, height));
    angle = angle;
    center = centroid;
    vertices.resize(4);
    rectangle.points(vertices.data());
    initializeContours();
}

/**
 * Checks is this box is inside of another box
 * @param other  : Other box
 * @return 0 / 1 : Boolean indicating if box1 is inside box2
 */
bool Box::isInsideOf(const Box &other) const
{
    // Check if all corners of box1 are inside box2
    for (const auto &vertex : vertices)
    {
        if (cv::pointPolygonTest(other.vertices, vertex, false) < 0)
        {
            return false;
        }
    }

    return true;
}

/**
 * Merges the box with another box
 * @param other      : Other box
 * @return mergedBox : Merged box
 */
Box Box::mergeWith(const Box &other) const
{
    std::vector<cv::Point2f> points;
    points.insert(points.end(), vertices.begin(), vertices.end());
    points.insert(points.end(), other.vertices.begin(), other.vertices.end());

    // Create a box that includes all points
    Box mergedBox{points};

    return mergedBox;
}

/**
 * Returns expanded box by a factor centered on its centroid
 * @param factor       : Expansion factor
 * @result expandedBox : Expanded box
 */
Box Box::expand(const float factor) const
{
    // Create a copy of the rectangle and expand its size
    cv::RotatedRect expandedRect = rectangle;
    expandedRect.size.width *= factor;
    expandedRect.size.height *= factor;

    // Extract the expanded vertices
    std::vector<cv::Point2f> expandedVertices{4};
    expandedRect.points(expandedVertices.data());

    Box expandedBox{Box(expandedVertices)};

    return expandedBox;
}

/**
 * Draws the box on an image
 * @param image : Image where it is desired to draw the box
 * @param color : Box color
 */
void Box::draw(cv::Mat &image, const cv::Scalar &color) const
{
    for (size_t i = 0; i < 4; ++i)
    {
        cv::line(image, vertices[i], vertices[(i + 1) % 4], color, 4);
    }
}

/**
 * Generates random points inside the box
 * @param n              : Number of random points to generate
 * @result sampledPoints : Random points inside the box
 */
std::vector<cv::Point2f> Box::getRandomPointsInside(const int n)
{
    std::vector<cv::Point2f> sampledPoints;
    cv::RNG rng(cv::getTickCount());
    for (int i = 0; i < n; ++i)
    {
        float xOffset = rng.uniform(-0.5f, 0.5f) * width;
        float yOffset = rng.uniform(-0.5f, 0.5f) * height;
        cv::Point2f sampledPoint =
            center + cv::Point2f(xOffset * std::cos(angle * CV_PI / 180) -
                                     yOffset * std::sin(angle * CV_PI / 180),
                                 xOffset * std::sin(angle * CV_PI / 180) +
                                     yOffset * std::cos(angle * CV_PI / 180));
        sampledPoints.push_back(sampledPoint);
    }

    return sampledPoints;
}

/**
 * Checks if an angle is close enough to another or its perpendiculars
 * @param avgAngle : Average angle
 * @param devAngle : Standard deviation
 * @return 0 / 1   : Boolean indicating if it is close or not
 */
bool Box::isAngleValid(const int avgAngle, const float devAngle) const
{
    return (std::abs(angle - avgAngle) <= devAngle) ||
           (std::abs(angle - (avgAngle + 90.0f)) <= devAngle) ||
           (std::abs(angle - (avgAngle - 90.0f)) <= devAngle) ||
           (std::abs(angle - (avgAngle - 180.0f)) <= devAngle);
}

/**
 * Checks if the area is close enough to another value
 * @param avgArea : Average area
 * @param devArea : Standard deviation
 * @return 0 / 1  : Boolean indicating if it is close or not
 */
bool Box::isAreaValid(const float avgArea, const float devArea) const
{
    return std::abs(area - avgArea) / avgArea <= devArea;
};

/**
 * Checks if the aspect ratio is close enough to another value
 * @param avgAspectRatio : Average area
 * @param devAspectRatio : Standard deviation
 * @return 0 / 1         : Boolean indicating if it is close or not
 */
bool Box::isAspectRatioValid(const float avgAspectRatio, const float devAspectRatio) const
{
    return std::abs(aspectRatio - avgAspectRatio) / avgAspectRatio <= devAspectRatio;
};

/**
 * Helper construcotr function to initialize the contours from the vertices
 */
void Box::initializeContours()
{
    contours.clear();
    std::vector<cv::Point> intVertices;
    for (const auto &vertex : vertices)
    {
        intVertices.emplace_back(cv::Point(cvRound(vertex.x), cvRound(vertex.y)));
    }
    contours.push_back(intVertices);
}
