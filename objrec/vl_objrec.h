#ifndef VL_OBJREC_H
#define VL_OBJREC_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <evaluation/benchmark.h>

namespace vislab
{
  namespace objrec
  {
    
    struct Detection;//declared at object_detection.h

    // Object detection
    int classifyObjectsSift( vislab::evaluation::ImgInfo& files_test, std::vector<vislab::evaluation::ImgInfo>& files_train, unsigned numClasses );
    int detectObjectsSift( vislab::evaluation::ImgInfo& files_test, std::vector<vislab::evaluation::ImgInfo>& files_train, unsigned numClasses );

    int classifyObjectsSiftNNFast( vislab::evaluation::ImgInfo& files_test, std::vector<vislab::evaluation::ImgInfo>& classes_train, std::vector<cv::flann::Index*> finders, const unsigned numClasses, const int knnSearchParamsChecks );
    int classifyObjectsSiftLocalNN( vislab::evaluation::ImgInfo& files_test, std::vector<int>& labels, cv::flann::Index* finder, const unsigned numClasses, const int knnSearchParamsChecks );
    int classifyObjectsSiftLocalNNSoft( vislab::evaluation::ImgInfo& files_test, std::vector<std::vector<float> >& classprobs, cv::flann::Index* finder, const unsigned numClasses, const int knnSearchParamsChecks );
    int detectObjectsSurfLocalNN( vislab::evaluation::ImgInfo& files_test, std::vector<int>& labels, cv::flann::Index* finder, std::vector<cv::KeyPoint> allpoints, std::vector<cv::Point2f> offsets, const unsigned numClasses, const int knnSearchParamsChecks );

    // Object detection
    int classifyObjectsGenericLocalNN( vislab::evaluation::ImgInfo& files_test, std::vector<int>& labels, cv::flann::Index* finder, const unsigned numClasses, const int knnSearchParamsChecks );
    int classifyObjectsGenericLocalNN( cv::Mat& descs, std::vector<int>& labels, cv::flann::Index* finder, const unsigned numClasses, const int knnSearchParamsChecks );
    int classifyObjectsGenericLocalNN( cv::Mat& descs, std::vector<int>& labels, cv::flann::Index* finder, const unsigned numClasses, const int knnSearchParamsChecks, std::vector<float>& probs );

    // Shape recognition
    int classifyObjectsSegmentLocalNN( vislab::evaluation::ImgInfo& files_test, std::vector<int>& labels, cv::flann::Index* finder, unsigned numClasses, const int knnSearchParamsChecks );

    // Evaluation for detection tasks (precision, recall, etc.)
    void evaluatedets( vislab::evaluation::ImgInfo& files_info, std::vector<Detection>& dets, int* truepos, int *falsepos, float thresh, int labelnum = 0 );
    void evaluatealldets( std::vector<vislab::evaluation::ImgInfo>& files_info, std::vector<std::vector<Detection> >& alldets, int labelnum );

    // Drawing detections for visualisation
    void drawdetections(cv::Mat &img, std::vector<Detection>& dets, std::vector<std::string>& objects, int width=1);
    void drawgt(cv::Mat &img, std::vector<cv::Rect>& dets, int width=1);


    // Different functions to compare descriptors
    bool sortpair(const std::pair<double, cv::KeyPoint>& lhs, const std::pair<double, cv::KeyPoint>& rhs);
    double kpdistance( const std::vector<cv::KeyPoint> &points1, const std::vector<cv::KeyPoint> &points2, double maxdist=10000 );
    double kpdescdistance( const std::vector<cv::KeyPoint> &points1,  const std::vector<cv::KeyPoint> &points2, 
                           const std::vector<cv::Mat> &descs1, const std::vector<cv::Mat> &descs2 );
    double kpsiftdescdistance( const std::vector<cv::KeyPoint> &points1,  const std::vector<cv::KeyPoint> &points2, 
                               const cv::Mat &descs1, const cv::Mat &descs2 );
    double kpcolhistdistance( const std::vector<cv::KeyPoint> &points1,  const std::vector<cv::KeyPoint> &points2, 
                              const std::vector<cv::MatND> &descs1, const std::vector<cv::MatND> &descs2 );
    void makegauss( cv::Mat &out, std::vector<cv::KeyPoint> &points);

  } //namespace objrec
}//namespace vislab

#endif // VL_OBJREC_H
