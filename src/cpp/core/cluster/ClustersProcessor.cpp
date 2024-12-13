#include "core/cluster/ClustersProcessor.h"

/**
 * Perform density-based clustering (DBSCAN) on a set of points.
 *
 * This function iteratively assigns points to clusters based on their
 * proximity, using the DBSCAN algorithm. Points are considered part
 * of a cluster if they have enough neighbors within the defined
 * distance threshold.
 *
 * @param points : A vector of 2D points to be clustered
 */
void ClustersProcessor::performClustering(const std::vector<cv::Point2f> &points)
{
    clusterLabels.assign(points.size(), -1);
    clusters.clear();
    int clusterCount{0};

    for (size_t i = 0; i < points.size(); ++i)
    {
        if (clusterLabels[i] != -1)
        {
            continue; // Skip already labeled points
        }

        // Find neighbors
        std::vector<int> neighbors = findNeighbors(points, i);
        if (neighbors.size() < minPoints)
        {
            continue; // Mark as noise
        }

        // Form a new cluster
        createCluster(points, neighbors, ++clusterCount);
    }
}

/**
 * Find the two closest clusters based on a combined distance and size score.
 *
 * This function identifies the pair of clusters that are closest to each other,
 * taking into account both the Euclidean distance between their centroids and
 * a size similarity score (log-scale difference in cluster sizes).
 *
 * @param minScore  : Minimum score of the closest clusters
 * @return clusterA : Index of the first cluster
 * @return clusterB : Index of the second cluster
 */
std::pair<int, int> ClustersProcessor::findClosestClusters(double &minScore) const
{
    int clusterA = -1, clusterB = -1;
    minScore = std::numeric_limits<double>::max();

    for (size_t i = 0; i < clusters.size(); ++i)
    {
        for (size_t j = i + 1; j < clusters.size(); ++j)
        {
            // Get centroids of clusters i and j
            cv::Point2f centroidA = clusters[i].getCentroid();
            cv::Point2f centroidB = clusters[j].getCentroid();

            // Calculate distance between centroids
            double distance = cv::norm(centroidA - centroidB);

            // Calculate size similarity score
            double sizeRatio = static_cast<double>(clusters[i].getSize()) / clusters[j].getSize();
            double sizeScore = std::abs(std::log(sizeRatio)); // Log scale for balance

            // Combine distance and size into a single score
            double score = distance + sizeScore;

            // Update the closest clusters if a smaller score is found
            if (score < minScore)
            {
                minScore = score;
                clusterA = i;
                clusterB = j;
            }
        }
    }

    return {clusterA, clusterB};
}

/**
 * Reduce the number of clusters by merging the closest clusters.
 *
 * This function iteratively reduces the total number of clusters until
 * it reaches the specified maximum. Clusters are merged based on a
 * distance and size similarity score.
 *
 * @param maxClusters : The desired maximum number of clusters
 */
void ClustersProcessor::reduceClusters(size_t maxClusters)
{
    while (clusters.size() > maxClusters)
    {
        double minScore;
        auto [clusterA, clusterB] = findClosestClusters(minScore);

        // Merge clusterB into clusterA
        clusters[clusterA].mergeWith(clusters[clusterB]);

        // Remove clusterB
        clusters.erase(clusters.begin() + clusterB);
    }
}

/**
 * Reassign points to the closest clusters iteratively.
 *
 * This function refines the clustering by calculating the centroids of all
 * clusters and reassigning each point to the nearest cluster. It is performed
 * for a specified number of iterations to improve cluster consistency.
 *
 * @param iterations : The number of iterations for reassigning points.
 */
void ClustersProcessor::reassignPointsToClosestClusters(const size_t iterations)
{
    for (size_t iter = 0; iter < iterations; ++iter)
    {
        // Calculate centroids for all clusters
        std::vector<cv::Point2f> centroids{clusters.size()};

        for (size_t i = 0; i < clusters.size(); ++i)
        {
            centroids[i] = clusters[i].getCentroid();
        }

        // Create a new cluster assignment map
        std::vector<Cluster> newClusters(clusters.size());

        // Reassign points to the nearest cluster
        for (const auto &cluster : clusters)
        {
            for (const auto &point : cluster.getPoints())
            {
                double minDistance = std::numeric_limits<double>::max();
                size_t closestCluster = 0;

                for (size_t j = 0; j < centroids.size(); ++j)
                {
                    double distance = cv::norm(point - centroids[j]);
                    if (distance < minDistance)
                    {
                        minDistance = distance;
                        closestCluster = j;
                    }
                }

                // Assign the point to the closest cluster
                newClusters[closestCluster].addPoint(point);
            }
        }

        // Remove empty clusters
        clusters.clear();
        for (const auto &cluster : newClusters)
        {
            if (cluster.getSize() > 0)
            {
                clusters.push_back(std::move(cluster));
            }
        }
    }
}

/**
 * Find the neighbors of a point within the distance threshold.
 *
 * For a given point, this function finds all other points in the dataset
 * that are within the specified distance threshold.
 *
 * @param points : A vector of 2D points
 * @param index  : The index of the point for which neighbors are being calculated
 * return        : A vector of indices representing the neighbors of the point.
 */
std::vector<int> ClustersProcessor::findNeighbors(const std::vector<cv::Point2f> &points,
                                                  size_t index)
{
    std::vector<int> neighbors;
    for (size_t j = 0; j < points.size(); ++j)
    {
        if (cv::norm(points[index] - points[j]) < distanceThreshold)
        {
            neighbors.push_back(j);
        }
    }
    return neighbors;
}

/**
 * Form a new cluster by assigning points to a cluster ID.
 *
 * This function takes a list of neighboring points and assigns them
 * to a new cluster, updating the cluster labels and creating a new
 * 'Cluster' object.
 *
 * @param points    : A vector of 2D points
 * @param neighbors : A vector of indices representing the neighbors
 * return clusterId : The unique ID of the new cluster
 */
void ClustersProcessor::createCluster(const std::vector<cv::Point2f> &points,
                                      const std::vector<int> &neighbors,
                                      int clusterId)
{
    Cluster currentClusterPoints;

    for (int idx : neighbors)
    {
        clusterLabels[idx] = clusterId;
        currentClusterPoints.addPoint(points[idx]);
    }

    clusters.push_back(currentClusterPoints);
}