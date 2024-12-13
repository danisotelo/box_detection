#include "core/cluster/Cluster.h"

/**
 * Merge another cluster into this cluster
 * @param other  : Other cluster
 */
void Cluster::mergeWith(const Cluster &other)
{
    points.insert(points.end(), other.points.begin(), other.points.end());
}

/**
 * Get the centroid of the cluster
 * @result centroid : Cluster centroid
 */
cv::Point2f Cluster::getCentroid() const
{
    cv::Point2f centroid(0, 0);
    for (const auto &point : points)
    {
        centroid += point;
    }
    if (!points.empty())
    {
        centroid.x /= points.size();
        centroid.y /= points.size();
    }

    return centroid;
}

/**
 * Draws the cluser on an image
 * @param image : Image where it is desired to draw the cluster points
 * @param color : Cluster color
 */
void Cluster::draw(cv::Mat &image, cv::Scalar &color) const
{
    for (const auto &point : points)
    {
        cv::circle(image, point, 3, color, -1);
    }
}

/**
 * Find the closest line of a box to the cluster
 * @param detectedBoxes : Detected boxes
 * @result closestStart : Start of the solution line
 * @result closestEnd   : End of the solution line
 */
std::pair<cv::Point2f, cv::Point2f> Cluster::findClosestLine(const std::vector<Box> &detectedBoxes) const
{
    double minDistance = std::numeric_limits<double>::max();
    cv::Point2f closestStart, closestEnd;
    cv::Point2f centroid = getCentroid();

    for (const auto &box : detectedBoxes)
    {
        for (int i = 0; i < 4; ++i)
        {
            cv::Point2f start = box.vertices[i];
            cv::Point2f end = box.vertices[(i + 1) % 4];

            // The distance is the sum of the squared distances between the centroid and each side of the segment
            double distance = pow(start.x - centroid.x, 2) + pow(start.y - centroid.y, 2) +
                              pow(end.x - centroid.x, 2) + pow(end.y - centroid.y, 2);

            if (distance < minDistance)
            {
                minDistance = distance;
                closestStart = start;
                closestEnd = end;
            }
        }
    }

    return {closestStart, closestEnd};
}

/**
 * Compute the projections and perpendicular distances of points onto a line
 * @param closestStart   : Line starting point
 * @param closestEnd     : Line ending point
 * @result projections   : Projections of the point cloud on the line
 * @result perpDistances : Normal distances between the line and the points
 */
std::pair<std::vector<std::pair<double, double>>, std::vector<double>>
Cluster::computeProjAndDist(const cv::Point2f &closestStart, const cv::Point2f &closestEnd) const
{
    std::vector<std::pair<double, double>> projections;
    std::vector<double> perpDistances;

    cv::Point2f lineVector = closestEnd - closestStart;
    double lineLength = cv::norm(lineVector);
    cv::Point2f lineUnitVector = lineVector * (1.0 / lineLength);

    cv::Point2f centroid = getCentroid();
    double centroidProjection = centroid.dot(lineUnitVector);

    for (const auto &point : points)
    {
        cv::Point2f relativePoint = point - closestStart;

        // Projection onto the line
        double projection = relativePoint.dot(lineUnitVector);
        double distanceToCentroidProj = std::abs(projection - centroidProjection);
        projections.emplace_back(projection, distanceToCentroidProj);

        // Perpendicular distance to the line
        double perpDistance = std::abs(relativePoint.x * lineUnitVector.y - relativePoint.y * lineUnitVector.x);
        perpDistances.emplace_back(perpDistance);
    }

    return {projections, perpDistances};
}

/**
 * Filter projections and compute box dimensions
 * @param projections      : Projections of the point cloud on the line
 * @param perpDistances    : Normal distances between the line and the points
 * @param devProjections   : Standard deviation for projections
 * @param devPerpDistances : Standard deviation for normal distances
 * @result boxDimensions   : Dimensions of the cluster box
 */
cv::Size2f Cluster::getBoxDim(
    const std::vector<std::pair<double, double>> &projections,
    const std::vector<double> &perpDistances, float devProjections, float devPerpDistances) const
{
    // Sort projections by distance to centroid projection
    std::vector<std::pair<double, double>> sortedProjections = projections;
    std::vector<double> sortedPerpDistances = perpDistances;

    std::sort(sortedProjections.begin(), sortedProjections.end(),
              [](const std::pair<double, double> &a, const std::pair<double, double> &b)
              {
                  return a.second < b.second;
              });

    std::sort(sortedPerpDistances.begin(), sortedPerpDistances.end());

    // Exclude outliers
    int lowerBoundIndex = static_cast<int>(devProjections * sortedProjections.size());
    int upperBoundIndex = static_cast<int>((1.0f - devProjections) * sortedProjections.size());
    int maxPerpDistIndex = static_cast<int>((1.0f - devPerpDistances) * sortedPerpDistances.size());

    double minProjection = sortedProjections[lowerBoundIndex].first;
    double maxProjection = sortedProjections[upperBoundIndex].first;
    double maxPerpDistance = sortedPerpDistances[maxPerpDistIndex];

    // Compute box dimensions
    cv::Size2f boxDimensions{static_cast<float>(maxProjection - minProjection),
                             static_cast<float>(maxPerpDistance)};

    return boxDimensions;
}