#ifndef CLUSTERS_PROCESSOR_H
#define CLUSTERS_PROCESSOR_H

#include "core/box/Box.h"
#include "core/cluster/Cluster.h"
#include "utils/colors.h"

class ClustersProcessor
{
public:
    ClustersProcessor(const float distanceThreshold, const int minPoints)
        : distanceThreshold(distanceThreshold), minPoints(minPoints) {};

    // Perform DBSCAN clustering
    void performClustering(const std::vector<cv::Point2f> &points);

    // Get clusters
    std::vector<Cluster> getClusters() { return clusters; }

    // Find the two closest clusters
    std::pair<int, int> findClosestClusters(double &minScore) const;

    // Reduce clusters to a maximum number
    void reduceClusters(size_t maxClusters);

    // Reassign point to the closest clusters
    void reassignPointsToClosestClusters(const size_t iterations);

private:
    const float distanceThreshold; // Distance threshold for clustering
    const int minPoints;           // Cluster minimum points

    std::vector<int> clusterLabels;
    std::vector<Cluster> clusters;

    std::vector<int> findNeighbors(const std::vector<cv::Point2f> &points, size_t index);

    void createCluster(const std::vector<cv::Point2f> &points,
                       const std::vector<int> &neighbors,
                       int clusterId);
};

#endif // CLUSTERS_PROCESSOR_H
