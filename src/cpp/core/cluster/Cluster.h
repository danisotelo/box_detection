#ifndef CLUSTER_H
#define CLUSTER_H

#include "core/box/Box.h"
#include <opencv2/core.hpp>

class Cluster
{
public:
    Cluster() {};

    // Add a point to the cluster
    void addPoint(const cv::Point2f &point) { points.push_back(point); }

    // Merge another cluster into this cluster
    void mergeWith(const Cluster &other);

    // Get the centroid of the cluster
    cv::Point2f getCentroid() const;

    // Get the size of the cluster
    size_t getSize() const { return points.size(); }

    // Get all points in the cluster
    const std::vector<cv::Point2f> &getPoints() const { return points; }

    // Draw the cluster on an image
    void draw(cv::Mat &image, cv::Scalar &color) const;

    // Find the closest line of a box to the cluster
    std::pair<cv::Point2f, cv::Point2f> findClosestLine(const std::vector<Box> &detectedBoxes) const;

    // Compute projections of points onto a line
    std::pair<std::vector<std::pair<double, double>>, std::vector<double>> computeProjAndDist(
        const cv::Point2f &closestStart, const cv::Point2f &closestEnd) const;

    // Filter projections and compute box dimensions
    cv::Size2f getBoxDim(
        const std::vector<std::pair<double, double>> &projections,
        const std::vector<double> &perpDistances, float devProjections, float devPerpDistances) const;

private:
    std::vector<cv::Point2f> points;
};

#endif // CLUSTERS_H