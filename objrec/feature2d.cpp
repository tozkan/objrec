#include "feature2d.h"
#include <keypoints/common.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdexcept>

namespace vislab
{
  namespace objrec
  {

    //default palette in lab colorspace
    unsigned char default_lab_palette_data[] = {0, 128, 128, 224, 42, 211, 154, 226, 67, 255, 128, 128, 82, 207, 20, 136, 208, 195, 248, 106, 223, 233, 80, 114};
    const cv::Mat default_lab_palette =  cv::Mat(8,3,CV_8UC3,(void*)default_lab_palette_data);


    BimpCVFeature2D::BimpCVFeature2D(const std::vector<double> lambdas_,
                                     const unsigned int norientations_,
                                     const bool scaling_,
                                     const std::string extractor_name_)
      : bimp_detector(new vislab::keypoints::BimpFeatureDetector(lambdas_, norientations_, scaling_)){

      cv::initModule_nonfree();//init nonfree

      desc_extractor = cv::DescriptorExtractor::create(extractor_name_);
      if (!desc_extractor)
        throw std::runtime_error("empty desc_extractor_");
    }

    BimpCVFeature2D::~BimpCVFeature2D() {}

    int BimpCVFeature2D::descriptorSize() const
    {
      return desc_extractor->descriptorSize();
    }

    int BimpCVFeature2D::descriptorType() const
    {
      return desc_extractor->descriptorType();
    }

    void BimpCVFeature2D::operator()(cv::InputArray _image, cv::InputArray _mask, std::vector<cv::KeyPoint>& keypoints) const
    {
      cv::Mat image = _image.getMat(), mask = _mask.getMat();
      if( image.type() != CV_8UC1 )
        cv::cvtColor(image, image, CV_BGR2GRAY);
      bimp_detector->detect( image, keypoints, mask );
    }

    void BimpCVFeature2D::operator()( cv::InputArray _image, cv::InputArray _mask, std::vector<cv::KeyPoint>& keypoints,
                                      cv::OutputArray _descriptors, bool useProvidedKeypoints ) const
    {
      cv::Mat image = _image.getMat(), mask = _mask.getMat();
      if( image.type() != CV_8UC1 )
        cv::cvtColor(image, image, CV_BGR2GRAY);
      if (!useProvidedKeypoints)
        (*this)( image, mask, keypoints );
      //calc descriptors
      _descriptors.create(keypoints.size(), desc_extractor->descriptorSize(), desc_extractor->descriptorType());
      cv::Mat descriptors = _descriptors.getMat();
      desc_extractor->compute( image, keypoints, descriptors );
    }

    void BimpCVFeature2D::computeImpl( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors ) const
    {
      (*this)(image, cv::Mat(), keypoints, descriptors, true);
    }

    void BimpCVFeature2D::detectImpl( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, const cv::Mat& mask ) const
    {
      (*this)(image, mask, keypoints);
    }

    cv::Mat& bioDescriptorExtractor(const cv::Mat& src, const std::vector<vislab::keypoints::KPData>& datas, const std::vector<cv::KeyPoint>& kpts, cv::Mat& descriptors){
      const int nbytes = BIO_DESCRIPTOR_SIZE;
      const int multiplier=1;//FIXME
      descriptors.create(kpts.size(), nbytes, CV_8UC1);

      std::map<int, int> lambdas;
      for (int i=0; i < (int)datas.size();i++){
        lambdas.insert(std::make_pair(datas[i].lambda,i));
      }

      std::vector<float> randomValues;
      const float nPixelsLayer[4] = {4, 6, 9, 13};
      const float nDistanceLayer[4] = {0.2, 0.45, 0.75, 1};

      const int nLayers = 4;
      float degrees = 2*3.1415/nPixelsLayer[0];

      for (int a = 0; a < nLayers; a++){
        for (int i = 0; i < nPixelsLayer[a]; i++){
          randomValues.push_back(nDistanceLayer[a]*cos(degrees));
          randomValues.push_back(nDistanceLayer[a]*sin(degrees));
          degrees = degrees + 2*3.1415/nPixelsLayer[a];
        }
      }

      #pragma omp parallel for schedule(dynamic,1) default(shared)
      for (int i = 0; i < (int)kpts.size();i++){
        uchar* data_out = descriptors.ptr<uchar>(i);
        for (int j = 0; j < nbytes*2; j = j + 2){
          int b = (int)floor((kpts[i].pt.y/(src.rows/datas[lambdas[kpts[i].size]].C_array[0].rows) + randomValues[j]*4*multiplier)+0.5);
          int a = (int)floor((kpts[i].pt.x/(src.cols/datas[lambdas[kpts[i].size]].C_array[0].cols) + randomValues[j+1]*4*multiplier)+0.5);
          uchar desc8bits = 0x00;
          //std::cout << a << " " << randomValues[j]*4 << " " << b << " " << randomValues[j+1]*4 << "\n";
          for (int h=0; h<8; h++){
            uchar aux = 0x01;
            const double* data_in = datas[lambdas[kpts[i].size]].C_array[h].ptr<double>(b);
            const double* data_in_center = datas[lambdas[kpts[i].size]].C_array[h].ptr<double>((int)kpts[i].pt.y/(src.rows/datas[lambdas[kpts[i].size]].C_array[0].rows));
            //std::cout << "i="<< i << " size=" << kpts[i].size << " ori=" << h << " val="<< (int)data_in[a] << " x=" << kpts[i].pt.x/(kpts[i].size/4) << " y=" << kpts[i].pt.y/(kpts[i].size/4) << " x_=" << a << " y_=" << b << " xr=" << randomValues[j] << " yr=" << randomValues[j+1] << " val_c=" << data_in_center[(int)kpts[i].pt.x] << " \n";
            if (data_in[a]>data_in_center[(int)(kpts[i].pt.x/(src.cols/datas[lambdas[kpts[i].size]].C_array[0].cols))]){
              desc8bits = desc8bits | (aux << h);
            }
          }
          //std::cout << "\n " << (std::bitset<8>) desc8bits << " " << i << " " << j << "\n";
          data_out[j/2] = desc8bits;
          //getchar();
        }
      }

      return descriptors;
    }

    cv::Mat& bioDescriptor_8U_to_32F(const cv::Mat& desc8u, cv::Mat& desc32f){
      CV_Assert(desc8u.type() == CV_8UC1);

      desc32f.create(desc8u.rows, BIO_DESCRIPTOR_SIZE*8, CV_32F);
      for (unsigned int row=0; row<desc8u.rows; row++){
        for(unsigned int col=0; col<BIO_DESCRIPTOR_SIZE; col++){
          const unsigned char desc8u_elem = desc8u.at<unsigned char>(row,col);
          desc32f.at<float>(row,col*8+0) = (desc8u_elem & 1) != 0;
          desc32f.at<float>(row,col*8+1) = (desc8u_elem & 2) != 0;
          desc32f.at<float>(row,col*8+2) = (desc8u_elem & 4) != 0;
          desc32f.at<float>(row,col*8+3) = (desc8u_elem & 8) != 0;
          desc32f.at<float>(row,col*8+4) = (desc8u_elem & 16) != 0;
          desc32f.at<float>(row,col*8+5) = (desc8u_elem & 32) != 0;
          desc32f.at<float>(row,col*8+6) = (desc8u_elem & 64) != 0;
          desc32f.at<float>(row,col*8+7) = (desc8u_elem & 128) != 0;
        }
      }
      return desc32f;
    }

    BimpBioFeature2D::BimpBioFeature2D( const std::vector<double> lambdas_,
                                        const unsigned int norientations_,
                                        const bool scaling_ )
      : bimp_detector(new vislab::keypoints::BimpFeatureDetector(lambdas_, norientations_, scaling_)),
        norientations(norientations_) {}

    BimpBioFeature2D::~BimpBioFeature2D() {}

    int BimpBioFeature2D::descriptorSize() const
    {
      return BIO_DESCRIPTOR_SIZE;
    }

    int BimpBioFeature2D::descriptorType() const
    {
      return CV_8UC1;
    }

    void BimpBioFeature2D::operator()(cv::InputArray _image, cv::InputArray _mask, std::vector<cv::KeyPoint>& keypoints) const
    {
      cv::Mat image = _image.getMat(), mask = _mask.getMat();
      if( image.type() != CV_8UC1 )
        cv::cvtColor(image, image, CV_BGR2GRAY);
      bimp_detector->detect( image, keypoints, mask );
    }

    void BimpBioFeature2D::operator()( cv::InputArray _image, cv::InputArray _mask, std::vector<cv::KeyPoint>& keypoints,
                                      cv::OutputArray _descriptors, bool useProvidedKeypoints ) const
    {
      cv::Mat image = _image.getMat(), mask = _mask.getMat();
      if( image.type() != CV_8UC1 )
        cv::cvtColor(image, image, CV_BGR2GRAY);

      std::vector<vislab::keypoints::KPData> kpdatas;
      if (useProvidedKeypoints){//calc gabor responses for provided keypoint lambdas
        std::vector<double> lambdas;
        for (unsigned int i=0; i<keypoints.size(); i++)
          lambdas.push_back(keypoints[i].size);
        std::sort(lambdas.begin(), lambdas.end());
        std::vector<double>::iterator new_end = std::unique(lambdas.begin(), lambdas.end());
        lambdas.resize(std::distance(lambdas.begin(),new_end));
        kpdatas.resize(lambdas.size());
        for (unsigned int i=0; i<lambdas.size(); i++)
          vislab::keypoints::gaborfilterbank(image, kpdatas[i].RO_array, kpdatas[i].RE_array,
                                             kpdatas[i].C_array, lambdas[i], this->norientations);
      }else//calc keypoints keeping gabor responses
        bimp_detector->detect( image, keypoints, kpdatas, mask );

      //calc descriptors
      cv::Mat descriptors = _descriptors.getMat();
      bioDescriptorExtractor( image, kpdatas, keypoints, descriptors );
    }

    void BimpBioFeature2D::operator()( cv::InputArray _image, cv::InputArray _mask, std::vector<cv::KeyPoint>& keypoints,
                                       std::vector<vislab::keypoints::KPData>& kpdatas, cv::OutputArray _descriptors, bool useProvidedKeypoints ) const
    {
      cv::Mat image = _image.getMat(), mask = _mask.getMat();
      if( image.type() != CV_8UC1 )
        cv::cvtColor(image, image, CV_BGR2GRAY);
      if (!useProvidedKeypoints)
        bimp_detector->detect( image, keypoints, kpdatas, mask );

      //calc descriptors
      cv::Mat descriptors = _descriptors.getMat();
      bioDescriptorExtractor( image, kpdatas, keypoints, descriptors );
    }

    void BimpBioFeature2D::computeImpl( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors ) const
    {
      (*this)(image, cv::Mat(), keypoints, descriptors, true);
    }

    void BimpBioFeature2D::detectImpl( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, const cv::Mat& mask ) const
    {
      (*this)(image, mask, keypoints);
    }

  }//namespace objrec
}//namespace vislab
