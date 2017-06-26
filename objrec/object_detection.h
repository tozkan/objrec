#ifndef OBJECT_DETECTION_H
#define OBJECT_DETECTION_H

#include "feature2d.h"
#include "vl_objrec.h" //old stuff
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>

namespace vislab
{
  namespace objrec
  {
    struct Detection
    {
      Detection(const cv::Point centre_ = cv::Point(0,0),
		const int labelnum_ = -1,
		const float strength_ = 0.0);
      
      cv::Point centre;
      int labelnum;
      float strength;
      bool correct;
      cv::Rect bbox;
      std::vector<cv::Point> mypoints;
      std::vector<cv::Point> hull;
    };
    

    struct ObjectDetectorParams{
      ObjectDetectorParams( const float vote_scale_ = 4,
			    const unsigned int kernel_size_ = 100,
			    const float alpha_ = 100.,
			    const float detection_thresh_ = -2e8,
			    const float overlap_thresh_ = 0.3,
			    const float merge_overlap_thresh_ = 0.8,
			    const unsigned int knn_search_neighbours_ = 21,
			    const unsigned int knn_search_checks_ = 100,
                            const cv::Ptr<cv::Feature2D> feature2d_detector_ = cv::Ptr<cv::Feature2D>(new BimpCVFeature2D()));
      
      float vote_scale;
      unsigned int kernel_size;
      float alpha;
      float detection_thresh;
      float overlap_thresh;//detection overlap threshold between different classes
      float merge_overlap_thresh;//merge overlap threshold for detections of same classes
      unsigned int knn_search_neighbours;
      unsigned int knn_search_checks;
      cv::Ptr<cv::Feature2D> feature2d_detector;
    };

    class ObjectDetector{
    public:
      ObjectDetector( const ObjectDetectorParams& params_ = ObjectDetectorParams() );
      ~ObjectDetector();

      void loadTrainingSet( const std::vector<std::string>& train_img_fnames,
			    const std::vector<int>& labels,
			    const std::vector<std::string>& class_names,
                            const bool resize = true,
                            const unsigned int resize_base = 300 );

      void setDetectionThresholds(const cv::Mat& thresholds);
      const cv::Mat& getDetectionThresholds() const;
      
      void detect(const cv::Mat& image, std::vector<Detection>& results, 
		  const bool resize = false,
		  const unsigned int resize_base = 300);
      void detect(const std::vector<cv::Mat>& images, std::vector<std::vector<Detection> >& results,
		  const bool resize = false,
		  const unsigned int resize_base = 300);

      const std::vector<std::string>& getClassNames() const;
      unsigned int getClassCount() const;
      const ObjectDetectorParams& getParams() const;
      
    private:
      struct onehypo;
      struct DetectorInternalState;
      struct PImpl;
      
      void detect1(const cv::Mat& image, std::vector<cv::Mat>& obj_cs, DetectorInternalState& state);//FIXME: can't be const because knnSearch
      void detect2(const std::vector<cv::Mat>& obj_cs, std::vector<Detection>& results, DetectorInternalState& state) const;
      void detect2s(const std::vector<cv::Mat>& obj_cs, std::vector<Detection>& results, DetectorInternalState& state) const;
      void removeOverlappingDetections(const std::vector<Detection>& input_detections, std::vector<Detection>& output_detections) const;

      cv::Ptr<PImpl> impl;
      const ObjectDetectorParams params;
      unsigned int n_classes;
      std::vector<std::string> class_names;
      cv::Mat detection_thresholds;
    };


    void drawdetections(cv::Mat &img, const std::vector<Detection>& dets, 
                        const std::vector<std::string>& objects, const int width=1);
    
  } //namespace objrec
}//namespace vislab

#endif //OBJECT_DETECTION_H
