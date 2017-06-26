#ifndef OBJREC_FEATURE2D_H
#define OBJREC_FEATURE2D_H

#include <vector>
#include <keypoints/keypoints.h>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

namespace vislab
{
  namespace objrec
  {
    class BimpCVFeature2D: public cv::Feature2D {
    public:
      explicit BimpCVFeature2D(const std::vector<double> lambdas_ = vislab::keypoints::makeLambdasLog(8, 64, 2),
                               const unsigned int norientations_=8, 
                               const bool scaling_ = true,
                               const std::string extractor_name_ = "SIFT");
      
      virtual ~BimpCVFeature2D();

      // returns the descriptor size in bytes
      int descriptorSize() const;
      // returns the descriptor type
      int descriptorType() const;

      // Compute the Bimp features on an image
      void operator()(cv::InputArray _image, cv::InputArray _mask, std::vector<cv::KeyPoint>& keypoints) const;
      
      // Compute the Bimp features and CV descriptors on an image
      void operator()( cv::InputArray _image, cv::InputArray _mask, std::vector<cv::KeyPoint>& keypoints,
                       cv::OutputArray descriptors, bool useProvidedKeypoints=false ) const;
    protected:
      void computeImpl( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors ) const;
      void detectImpl( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, const cv::Mat& mask=cv::Mat() ) const;
    private:
      cv::Ptr<vislab::keypoints::BimpFeatureDetector> bimp_detector;
      cv::Ptr<cv::DescriptorExtractor> desc_extractor;
    };

    
    //FIXME: choose a better name bio->??
    const int BIO_DESCRIPTOR_SIZE = 32;

    cv::Mat& bioDescriptorExtractor(const cv::Mat& src, const std::vector<vislab::keypoints::KPData>& datas, const std::vector<cv::KeyPoint>& kpts, cv::Mat& descriptors);

    cv::Mat& bioDescriptor_8U_to_32F(const cv::Mat& desc8u, cv::Mat& desc32f);
    
    class BimpBioFeature2D: public cv::Feature2D {
    public:
      explicit BimpBioFeature2D( const std::vector<double> lambdas_ = vislab::keypoints::makeLambdasLog(8, 64, 2),
                                 const unsigned int norientations_=8, 
                                 const bool scaling_ = true );
      
      virtual ~BimpBioFeature2D();

      // returns the descriptor size in bytes
      int descriptorSize() const;
      // returns the descriptor type
      int descriptorType() const;

      // Compute the Bimp features on an image
      void operator()(cv::InputArray _image, cv::InputArray _mask, std::vector<cv::KeyPoint>& keypoints) const;
      
      // Compute the Bimp features and bio descriptors on an image
      void operator()( cv::InputArray _image, cv::InputArray _mask, std::vector<cv::KeyPoint>& keypoints,
                       cv::OutputArray _descriptors, bool useProvidedKeypoints=false ) const;

      // Compute the Bimp features and bio descriptors on an image with intermediate data
      void operator()( cv::InputArray _image, cv::InputArray _mask, std::vector<cv::KeyPoint>& keypoints,
                       std::vector<vislab::keypoints::KPData>& kpdatas, cv::OutputArray _descriptors, bool useProvidedKeypoints=false ) const;
    protected:
      void computeImpl( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors ) const;
      void detectImpl( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, const cv::Mat& mask=cv::Mat() ) const;
    private:
      const cv::Ptr<vislab::keypoints::BimpFeatureDetector> bimp_detector;
      const unsigned int norientations;
    };
    
  }//namespace objrec
}//namespace vislab

#endif //OBJREC_FEATURE2D_H
