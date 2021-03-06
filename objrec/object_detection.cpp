#include "object_detection.h"
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

#define PROFILE 0
#define DEBUG 1
#include <util/debug.h>

namespace vislab
{
  namespace objrec
  {

    // struct ExtDetection{
    //   ExtDetection();
    //   std::vector<float> pt_scales;
    //   std::vector<float> pt_scale_counts;
    //   float scale_sum;
    //   float scale_count;
    //   float scale_mean;
    //   float scale_median;
    // };

    //ExtDetection::ExtDetection()
    //  : scale_sum(0.0), scale_count(0.0), scale_mean(0.0) {}


    Detection::Detection(const cv::Point centre_,
                         const int labelnum_,
                         const float strength_)
      : centre(centre_), labelnum(labelnum_),
	strength(strength_), correct(false) {}

    ObjectDetectorParams::ObjectDetectorParams(const float vote_scale_,
                                               const unsigned int kernel_size_,
                                               const float alpha_,
                                               const float detection_thresh_,
                                               const float overlap_thresh_,
                                               const float merge_overlap_thresh_,
                                               const unsigned int knn_search_neighbours_,
                                               const unsigned int knn_search_checks_,
                                               const cv::Ptr<cv::Feature2D> feature2d_detector_)
      : vote_scale(vote_scale_), kernel_size(kernel_size_), alpha(alpha_),
        detection_thresh(detection_thresh_),
        overlap_thresh(overlap_thresh_), merge_overlap_thresh(merge_overlap_thresh_),
        knn_search_neighbours(knn_search_neighbours_),
        knn_search_checks(knn_search_checks_),
        feature2d_detector(feature2d_detector_) {}

    struct ObjectDetector::onehypo { //name courtesy of KT :D
      onehypo() {}

      int mypt;
      int nearestpt;
      float contribution;
      float distb;
      cv::Point2f offset;
      cv::Rect ROI;
    };

    struct ObjectDetector::DetectorInternalState{
      DetectorInternalState() {};

      std::vector<cv::Mat> obj_centre_vcount;
      std::vector<cv::Mat> obj_centre_sdist;
      std::vector<cv::Mat> obj_centre_kplists;
      std::vector<cv::Mat> obj_centre_scale;
      std::vector<cv::Mat> obj_centre_scalecount;
      //std::vector<cv::Mat> obj_centre_scale_mean;
      std::vector<std::vector<onehypo> > pt_store;
      std::vector<cv::KeyPoint> detect_kps;
    };

    struct ObjectDetector::PImpl{
      PImpl() {}

      //alllabels allows to recover label for each descriptor, point and offset
      std::vector<int> alllabels;
      cv::Mat alldescs;
      std::vector<cv::KeyPoint> allpoints;
      std::vector<cv::Point2f> alloffsets;
      std::vector<cv::Rect> allrects;
      std::vector<cv::Rect> medianrects_by_class;
      cv::Ptr<cv::flann::Index> alldescs_index;
    };

    ObjectDetector::ObjectDetector( const ObjectDetectorParams& params_ )
      :impl(new PImpl()), params(params_) {}

    ObjectDetector::~ObjectDetector(){}

    void ObjectDetector::loadTrainingSet( const std::vector<std::string>& train_img_fnames,
                                          const std::vector<int>& labels,
                                          const std::vector<std::string>& class_names,
                                          const bool resize,
                                          const unsigned int resize_base )
    {
      assert(train_img_fnames.size() == labels.size() && "train_img_fnames and labels should be the same size");
      this->n_classes = class_names.size();
      assert(n_classes > 0 && "number of classes must be greater than 0");

      std::vector< std::vector<int> > widths_by_class(n_classes);
      std::vector< std::vector<int> > heights_by_class(n_classes);

      impl->alldescs.create(0, params.feature2d_detector->descriptorSize(), params.feature2d_detector->descriptorType());
      DEBUG_MESSAGE("Extracting features from training images ");
      #pragma omp parallel for schedule(dynamic,1) default(shared)
      for (int i=0; i<train_img_fnames.size(); i++){
	cv::Mat train_img(cv::imread(train_img_fnames[i]));//load RGB
    if(train_img.cols == 0 || train_img.rows == 0) continue;
    if (resize){
	  float scale;
	  scale = (float)resize_base / (float)std::max(train_img.cols,train_img.rows);
	  cv::resize(train_img, train_img, cv::Size(), scale, scale, cv::INTER_CUBIC );
	}

	std::vector<cv::KeyPoint > points;
	std::vector<vislab::keypoints::KPData > datas;
	cv::Mat descriptors;
	std::vector<cv::Point2f> offsets;

	(*params.feature2d_detector)(train_img, cv::noArray(), points, descriptors);

	//cacl offsets
	offsets.resize(points.size());
	for(unsigned int p=0; p<points.size(); p++){
	  //cacl offsets
	  offsets[p] = cv::Point2f( (train_img.cols/2. - points[p].pt.x)/points[p].size,
                                    (train_img.rows/2. - points[p].pt.y)/points[p].size );
	}


	//gather points to calc bounding box
	std::vector<cv::Point> tmp_points;
	for (unsigned int p=0; p<points.size(); p++)
	  tmp_points.push_back(points[p].pt);
	cv::Rect bbox = cv::boundingRect(tmp_points);

        #pragma omp critical
	{
	  //push_back label, descriptors, points, offsets and rects
	  for (int r=0; r<descriptors.rows; r++){
	    impl->alllabels.push_back(labels[i]);
	    impl->alldescs.push_back(descriptors.row(r));
	    impl->allpoints.push_back(points[r]);
	    impl->alloffsets.push_back(offsets[r]);
	    impl->allrects.push_back(cv::Rect(0,0,train_img.cols/points[r].size,
					      train_img.rows/points[r].size));
	  }
	  //push back width/height by class
	  widths_by_class[labels[i]].push_back(bbox.width);
	  heights_by_class[labels[i]].push_back(bbox.height);
	  DEBUG_MESSAGE_RAW("Loading training image: " << i+1 << " out of " << train_img_fnames.size() << "\r" << std::flush);
	}
      }
      DEBUG_MESSAGE("Training done.");
      for (int i=0; i < n_classes; i++){
        std::sort(widths_by_class[i].begin(), widths_by_class[i].end());
        std::sort(heights_by_class[i].begin(), heights_by_class[i].end());
        impl->medianrects_by_class.push_back(cv::Rect(0,0,
                                                      widths_by_class[i][widths_by_class[i].size()/2],
                                                      heights_by_class[i][heights_by_class[i].size()/2]));
        // std::cout << "widths_by_class[" << i << "]=";
        // std::copy(widths_by_class[i].begin(), widths_by_class[i].end(), std::ostream_iterator<int>(std::cout, " "));
        // std::cout << std::endl;
        // std::cout << "heights_by_class[" << i << "]=";
        // std::copy(heights_by_class[i].begin(), heights_by_class[i].end(), std::ostream_iterator<int>(std::cout, " "));
        // std::cout << std::endl;
        // std::cout << "impl->medianrects_by_class[" << i << "]=" << impl->medianrects_by_class[i] << std::endl;
      }

      DEBUG_MESSAGE("n_classes=" << n_classes);
      DEBUG_MESSAGE("done");
      DEBUG_MESSAGE("alldescs.rows=" << impl->alldescs.rows);
      DEBUG_MESSAGE("Creating K-D tree... ");
      assert(impl->alldescs.rows > 0 && "empty descriptors");
      impl->alldescs_index = cv::Ptr<cv::flann::Index>(new cv::flann::Index(impl->alldescs, cv::flann::KDTreeIndexParams(4), cvflann::FLANN_DIST_L1));
      this->detection_thresholds.create(1,n_classes,CV_32FC1);
    }

    void ObjectDetector::setDetectionThresholds(const cv::Mat& thresholds)
    {
      this->detection_thresholds = -1.0 * thresholds;
      assert(this->detection_thresholds.cols == this->n_classes  && "detection_thresholds must have n_classes size");
    }

    void ObjectDetector::detect(const cv::Mat& image, std::vector<Detection>& results,
				const bool resize, const unsigned int resize_base){
      cv::Mat my_image(image);
      if (resize){
	float scale;
	scale = (float)resize_base / (float)std::max(image.cols, image.rows);
	cv::resize(my_image, my_image, cv::Size(), scale, scale, cv::INTER_CUBIC );
      }
      std::vector<cv::Mat> obj_cs;
      DetectorInternalState state;
      detect1(my_image, obj_cs, state);
      detect2(obj_cs, results, state);
      filterDetections(results);
    }

    void ObjectDetector::detect(const std::vector<cv::Mat>& images, std::vector<std::vector<Detection> >& results,
				const bool resize, const unsigned int resize_base){
      results.resize(images.size());
      for (unsigned int i=0; i<images.size(); i++)
        detect(images[i], results[i], resize, resize_base);
    }

    void ObjectDetector::detect1(const cv::Mat& image, std::vector<cv::Mat>& obj_cs, DetectorInternalState& state){
      cv::Mat descriptors;

      //DEBUG_MESSAGE("Extracting features from input image");
      state.detect_kps.clear();
      (*params.feature2d_detector)(image, cv::noArray(), state.detect_kps, descriptors);

      //DEBUG_MESSAGE("Creating vcount, sdist and kplists for each class");
      state.obj_centre_vcount.clear();
      state.obj_centre_sdist.clear();
      state.obj_centre_kplists.clear();
      state.obj_centre_scale.clear();
      state.obj_centre_scalecount.clear();
      state.pt_store.clear();
      int votes_rows = image.rows/params.vote_scale;
      int votes_cols = image.cols/params.vote_scale;
      for(unsigned int tmpi=0; tmpi<n_classes; tmpi++){
	state.obj_centre_vcount.push_back(cv::Mat::zeros(votes_rows, votes_cols, CV_32F));
	state.obj_centre_sdist.push_back(cv::Mat::zeros(votes_rows, votes_cols, CV_32F));
	state.obj_centre_kplists.push_back(cv::Mat::zeros(votes_rows, votes_cols, CV_32S)-1);
	state.obj_centre_scale.push_back(cv::Mat::zeros(votes_rows, votes_cols, CV_32F));
	state.obj_centre_scalecount.push_back(cv::Mat::zeros(votes_rows, votes_cols, CV_32F));
      }

      //DEBUG_MESSAGE("Find out nearest neighbour for each point");
      for(int d=0; d<state.detect_kps.size(); d++){
	std::vector<int>   knnin(params.knn_search_neighbours);
	std::vector<float> knndis(params.knn_search_neighbours);
	std::vector<float> query = cv::Mat_<float>(descriptors.row(d));
	impl->alldescs_index->knnSearch( query, knnin, knndis, params.knn_search_neighbours, cv::flann::SearchParams(params.knn_search_checks));

	int curx = state.detect_kps[d].pt.x;
	int cury = state.detect_kps[d].pt.y;
	float lambda_d = state.detect_kps[d].size;

	//std::vector<float> vs(n_classes,0);

	float distb = knndis[params.knn_search_neighbours-1];
	distb *= distb;
	std::vector<bool> classused(n_classes, false);

	for(unsigned n=0; n<params.knn_search_neighbours-1; n++){
	  int curlabel = impl->alllabels[knnin[n]];
	  float lambda_e = impl->allpoints[knnin[n]].size;
	  if(classused[curlabel]) continue;

	  float val = knndis[n];
	  // val = val*val; // + 1*dist*dist;

	  classused[curlabel] = true;

	  cv::Point2f offset = impl->alloffsets[knnin[n]] * lambda_d;
	  int votex = curx + offset.x;
	  int votey = cury + offset.y;

	  votex /= params.vote_scale;
	  votey /= params.vote_scale;

	  if(votex >= 0 && votex < state.obj_centre_sdist[curlabel].cols &&
	     votey >= 0 && votey < state.obj_centre_sdist[curlabel].rows){
	    state.obj_centre_sdist[curlabel].at<float>(votey,votex) += val-distb;
	    state.obj_centre_vcount[curlabel].at<float>(votey,votex) += 1;

	    int curkplist_idx = state.obj_centre_kplists[curlabel].at<int>(votey,votex);
	    onehypo temphypo;
	    temphypo.nearestpt = knnin[n];
	    temphypo.mypt = d;
      temphypo.distb = distb;
      temphypo.contribution = val - distb;
	    temphypo.ROI = impl->allrects[knnin[n]];
	    temphypo.ROI.width  *= state.detect_kps[d].size;
	    temphypo.ROI.height *= state.detect_kps[d].size;

	    if(curkplist_idx == -1){//init kplist index
	      std::vector<onehypo> tempvec;
	      tempvec.push_back(temphypo);
	      state.pt_store.push_back(tempvec);
	      state.obj_centre_kplists[curlabel].at<int>(votey,votex) = state.pt_store.size()-1;
	    }else
	      state.pt_store[curkplist_idx].push_back(temphypo);

	    state.obj_centre_scale[curlabel].at<float>(votey,votex) += (lambda_d / lambda_e);
	    state.obj_centre_scalecount[curlabel].at<float>(votey,votex) += 1;
	  }

	}
      }

      //DEBUG_MESSAGE("Create kernels");
      //create kernels for filtering
      int ksize = params.kernel_size / params.vote_scale + 1;
      cv::Mat kernelx_mask = cv::Mat::ones(ksize, 1, CV_32F);
      cv::Mat kernely_mask = cv::Mat::ones(ksize, 1, CV_32F);

      cv::Mat kernelx_dist = cv::Mat::ones(ksize, 1, CV_32F);
      cv::Mat kernely_dist = cv::Mat::ones(ksize, 1, CV_32F);

      for(int p=0; p<ksize; p++){
	kernelx_dist.at<float>(p, 0) = abs(p-ksize/2) * params.vote_scale;
	kernely_dist.at<float>(p, 0) = abs(p-ksize/2) * params.vote_scale;
      }

      //DEBUG_MESSAGE("Apply filters");
      obj_cs.resize(n_classes);
      for(unsigned int m=0; m<n_classes; m++){
	cv::Mat objc_sd = state.obj_centre_sdist[m];
	cv::Mat objc_vc = state.obj_centre_vcount[m];
	cv::Mat dst_sd, dst_vc;

	// Filter with a Gaussian and find maximum
	// cv::GaussianBlur(objc, dst, cv::Size(), 10);
	// filter2D( objc, dst, objc.depth(), kernel1);
	sepFilter2D( objc_sd, dst_sd, objc_sd.depth(), kernelx_mask, kernely_mask, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);
  sepFilter2D( objc_vc, dst_vc, objc_vc.depth(), kernelx_dist, kernely_dist, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);
  obj_cs[m] = dst_sd + dst_vc * params.alpha;
      }
    }


    void ObjectDetector::detect2(const std::vector<cv::Mat>& obj_cs, std::vector<Detection>& results, DetectorInternalState& state) const{
      assert(this->detection_thresholds.cols == this->n_classes && "detection_thresholds must have n_classes size");

      std::vector<Detection> detections;

      //DEBUG_MESSAGE("Finding detections...");
      int ksize = params.kernel_size / params.vote_scale + 1;

      for (unsigned int m=0; m<obj_cs.size(); m++){
	const cv::Mat obj_c = obj_cs[m];
	cv::Mat_<float> obj_c_sm;
	cv::GaussianBlur(obj_c, obj_c_sm , cv::Size(), 5);
        // do this until d1
        // see what happened
        // push the detections if it's ok, don't push if it isn't

	cv::Mat_<float> obj_c_dil(obj_c.size()), minlocarray;
	cv::Matx<uchar,3,3> element(1,1,1,1,0,1,1,1,1);
	cv::dilate(-obj_c_sm,obj_c_dil,element);
	minlocarray = -obj_c_sm - obj_c_dil;
	//imwrite("minloc.png", minlocarray);

	//float globmin, globmax;
	//cv::Point globminloc, globmaxloc;
	//cv::minMaxLoc(obj_c_sm, &globmin, &globmax, &globminloc, &globmaxloc);
  float min;
	for(int _x=0; _x<minlocarray.cols; _x++ )
	  for(int _y=0; _y<minlocarray.rows; _y++ ){
	    if(minlocarray.at<float>(_y,_x) <= 0) continue;
	    cv::Point minloc = cv::Point(_x,_y);
	    min = obj_c.at<float>(_y,_x);


	    // cv::Rect roi( minloc.x*params.vote_scale-50, minloc.y*params.vote_scale-20, 100, 40);
	    // std::cout << "minloc for " << m << " = " << minloc << std::endl;

	    // Collect all the individual points which voted for this hypothesis
	    std::vector<cv::Point> _mypoints;
	  //  std::vector<int> _ptindices;
      std::vector<float> _contributions, _distb;
	    std::vector<int> widths, heights;
	    for(int indy=minloc.y-ksize/2; indy<minloc.y+ksize/2; indy++){
	      for(int indx=minloc.x-ksize/2; indx<minloc.x+ksize/2; indx++){
		if(indy<0 || indy>=state.obj_centre_kplists[m].rows || indx<0 || indx>=state.obj_centre_kplists[m].cols) continue;
		int curind = state.obj_centre_kplists[m].at<int>(indy,indx);

		if(curind > -1 && curind<state.pt_store.size()){
		  for(unsigned ptind=0; ptind<state.pt_store[curind].size(); ptind++){
		//    _ptindices.push_back(state.pt_store[curind][ptind].mypt);
        // std::cout << state.detect_kps.size() << " " << state.pt_store[curind][ptind].mypt <<  std::endl;
        cv::Point2f curpt = state.detect_kps[state.pt_store[curind][ptind].mypt].pt;
        _contributions.push_back(state.pt_store[curind][ptind].contribution);
        _distb.push_back(state.pt_store[curind][ptind].distb);
        _mypoints.push_back(curpt);
        widths.push_back(state.pt_store[curind][ptind].ROI.width);
        heights.push_back(state.pt_store[curind][ptind].ROI.height);
		}
	      }
	    }



	     if(widths.size() == 0) continue;


      //std::vector<int> distances, distancessorted;
// for(int ptind=0; ptind<_ptindices.size(); ptind++){
//   // cv::Point2f curpt = state.detect_kps[_ptindices[ptind]].pt;
//   // _mypoints.push_back(curpt);
//   //int curdist = abs(curpt.x-minloc.x*params.vote_scale)+abs(curpt.y-minloc.y*params.vote_scale);
//   //distances.push_back(curdist);
//   //distancessorted.push_back(curdist);
// }
//sort(distancessorted.begin(),distancessorted.end());
// std::cout << distancessorted.size() << std::endl;
//float mediandist = distancessorted[distancessorted.size()/2];


	    std::sort( widths.begin(), widths.end() );
	    std::sort( heights.begin(), heights.end() );
	    int medianw = widths[widths.size()/2];
	    int medianh = heights[heights.size()/2];


	    Detection det;
	    det.strength = -min;
	    det.centre = minloc * params.vote_scale;
	    // Rect _bbox = boundingRect(_mypoints);
	    cv::Rect _bbox( det.centre.x - medianw/2, det.centre.y - medianh/2, medianw, medianh );

	    det.bbox = _bbox; //boundingRect(bbpoints);
	    det.labelnum = m;
	    det.mypoints = _mypoints;
      det.contributions = _contributions;
      det.widths = widths;
      det.heights = heights;
      det.distb=_distb;





    //    std::cout << m << " of " << this->detection_thresholds.cols << std::endl;
	    const float d_thresh = (this->detection_thresholds.at<float>(m) == 0) ? params.detection_thresh: this->detection_thresholds.at<float>(m);
	    if(min < d_thresh) detections.push_back(det);
	    // detections.push_back(det);
	  }
      }
      removeOverlappingDetections(detections, results);

    }
  }


    void ObjectDetector::removeOverlappingDetections(const std::vector<Detection>& input_detections, std::vector<Detection>& output_detections) const{
      // Eliminate overlapping detections
      std::vector<std::pair<float,int> > allboxes;
      std::vector<bool> remaining(input_detections.size(), true);
      for(unsigned i=0; i<input_detections.size(); i++)
        allboxes.push_back(std::make_pair(input_detections[i].strength, i));

      std::sort( allboxes.begin(), allboxes.end() );
      std::reverse( allboxes.begin(), allboxes.end() );

      // std::cout << detections.size() << " " << allboxes.size() << std::endl;
      for(int d=0; d<int(allboxes.size())-1; d++)
	{
	  Detection curdet = input_detections[allboxes[d].second];
	  cv::Rect dbox = curdet.bbox;

	  for(unsigned int c=d+1; c<allboxes.size(); c++){
	    if(remaining[c] == false) continue;
	    Detection testdet = input_detections[allboxes[c].second];
	    //if (curdet.labelnum != testdet.labelnum) continue;

	    cv::Rect cbox = testdet.bbox;

	    cv::Rect _interbox = dbox & cbox;
	    float _inter = _interbox.area();
	    float _union = cbox.area() + dbox.area() - _inter;

	    if(_inter/_union > params.overlap_thresh ) remaining[c] = false;

	    //check if same class boxes are (almost) contained in themselfs
	    if(curdet.labelnum == testdet.labelnum){
	      if(( (_inter/cbox.area()) > params.merge_overlap_thresh && cbox.area()<dbox.area() ) || //c merge_overlap_thresh inside d
		 ( (_inter/dbox.area()) > params.merge_overlap_thresh && dbox.area()<cbox.area() ) ) //d merge_overlap_thresh inside c
		remaining[c] = false; // merge
	    }

	    // bool contains = false;
	    // if(dbox.contains(cbox.tl()) && dbox.contains(cbox.br()) ) contains = true;
	    // if(cbox.contains(dbox.tl()) && cbox.contains(dbox.br()) ) contains = true;
	    // if(_inter/cbox.area()>0.8 && cbox.area()<dbox.area()) contains = true;
	    // if(_inter/dbox.area()>0.8 && dbox.area()<cbox.area()) contains = true;
	    // if(curdet.labelnum == testdet.labelnum && contains) remaining[c] = false;
	  }
	}
      // std::cout << remaining.size() << std::endl;
      output_detections.clear();
      for(unsigned d=0; d<allboxes.size(); d++)
	if(remaining[d])
	  output_detections.push_back(input_detections[allboxes[d].second]);
    }

    const std::vector<std::string>& ObjectDetector::getClassNames() const{
      return this->class_names;
    }

    unsigned int ObjectDetector::getClassCount() const{
      return this->n_classes;
    }

    const ObjectDetectorParams& ObjectDetector::getParams() const{
      return this->params;
    }

    const cv::Mat& ObjectDetector::getDetectionThresholds() const{
      return this->detection_thresholds;
    }

    void drawdetections(cv::Mat &img, const std::vector<Detection>& dets,
                        const std::vector<std::string>& objects, const int width){
      for(unsigned r=0; r<dets.size(); r++)
        {
          // if(dets[r].labelnum == 0) continue;
          int labelnum2 = dets[r].labelnum+1;
          // Scalar colour = Scalar(rand()%256,rand()%256,rand()%256);
          // cv::Scalar colour = cv::Scalar(64,255,64);
          cv::Scalar colour = cv::Scalar((labelnum2&1)*255,(labelnum2&2)*255,(labelnum2&4)*255);
          // if(dets[r].correct == false) colour = Scalar(64,64,255);

          std::string objname = objects[dets[r].labelnum];

          // Bounding Box
          cv::rectangle(img, dets[r].bbox, cv::Scalar(255,255,255),width*2);
          // rectangle(img, dets[r].bbox, Scalar(0,0,0),1);
          cv::rectangle(img, dets[r].bbox, colour, width);

          // Convex Hull
          // std::vector<std::vector<Point> > pState
          // Object Centre
          // circle( img, dets[r].centre, 4, Scalar(255,255,255), -1 );
          cv::circle( img, dets[r].centre, 4, colour, -1 );

          // Points
          // for(unsigned p=0; p<dets[r].mypoints.size(); p++)
          // {
          //     circle( img, dets[r].mypoints[p], 4, colour );
          // }

          // Object Class
          cv::Point textpt = dets[r].bbox.tl();
          cv::putText(img, objname, textpt, 1, width, colour);
        }
    }
    int count=0;
    void ObjectDetector::filterDetections(std::vector<Detection> detections) {

      count ++;
      std::ofstream myfile;

      for(int i=0; i<detections.size(); i++) {
        myfile.clear();
        myfile.open (std::to_string(count) + "results" + std::to_string(i) + ".txt");

        std::vector<cv::Point> points = detections[i].mypoints;
        std::vector<float> contributions = detections[i].contributions;
        std::vector<float> distb = detections[i].distb;

        // SORTING
        std::vector<std::pair<cv::Point,float> > order(points.size());
        for (int l=0; l<points.size(); l++){
          order[l] = std::make_pair(points[l], contributions[l]);
        }
        std::sort(order.begin(), order.end(),  [=](std::pair<cv::Point,float> a , std::pair<cv::Point,float> b) {
          return cv::norm(a.first - detections[i].centre) < cv::norm(b.first - detections[i].centre);
        });

        // SORTING
        std::vector<std::pair<cv::Point,float> > orderdistb(points.size());
        for (int l=0; l<points.size(); l++){
          orderdistb[l] = std::make_pair(points[l], distb[l]);
        }
        std::sort(orderdistb.begin(), orderdistb.end(),  [=](std::pair<cv::Point,float> a , std::pair<cv::Point,float> b) {
          return cv::norm(a.first - detections[i].centre) < cv::norm(b.first - detections[i].centre);
        });

        //  1ST METHOD : Create the 3 points needed for the parabola fitting
        float radius,threshold,min;
        std::vector<float> x,y;
        // 1st point : the last point
        radius = cv::norm(order[points.size()-2].first-detections[i].centre);
        x.push_back(radius);
        threshold = -orderdistb[points.size()-1].second - 8*pow(10,3)*M_PI*pow(radius,2);
        min = -detections[i].strength - order[points.size()-1].second;
        y.push_back(threshold-min);
        // std::cout << "YO " << radius << "  " << threshold-min << std::endl;
        // 2nd point : the first point
        radius = 0;
        x.push_back(radius);
        threshold = -orderdistb[0].second - 8*pow(10,3)*M_PI*pow(radius,2);
        float sum = 0;
        for(int j=0;j<points.size();j++) {
          sum += order[j].second;
          }
        min =  -detections[i].strength - sum ;
        y.push_back(threshold-min);
        // 3rd point : a point somewhere between the two other ones ( middle for the moment)
        radius = cv::norm(order[floor(points.size()/2)].first-detections[i].centre);
        x.push_back(radius);
        threshold = -orderdistb[floor(points.size()/2)+1].second - 8*pow(10,3)*M_PI*pow(radius,2);
        min = -detections[i].strength;
        for(int j=floor(points.size()/2)+1;j<points.size();j++) {
          min -= order[j].second;
        }
        y.push_back(threshold-min);
        float optimalRadius = calculateOptimum(x,y,2,3).at(0);
        float optimum = calculateOptimum(x,y,2,3).at(1);
        // std::cout << "Detection number " << i << " Image number " << count << " 1st method " << optimalRadius << " " << optimum << std::endl;

        // 2nd METHOD : Exhaustive search
        min = -detections[i].strength;
        optimum = -FLT_MAX;
        for(int j=0; j<points.size();j++) {
          float distanceToCentre = cv::norm(order[points.size()-j-2].first - detections[i].centre);
          float newRadius;
          if(j==points.size()-1) {
            newRadius=0;
          }
          else {
            newRadius = distanceToCentre;
          }
          float contribution = order[points.size()-j-1].second;
          min -= contribution;
          float threshold = -orderdistb[points.size()-j-1].second - 8*pow(10,3)*M_PI*pow(newRadius,2);
          // std::cout<< newRadius << "  " << threshold-min << std::endl;
        if(threshold-min>optimum) {
          optimum = threshold-min;
          optimalRadius = newRadius;
        }
        myfile << newRadius  << " " << threshold << " " << min << " " << threshold-min << " " << min/threshold << std::endl;
      }
      // std::cout << "Detection number " << i << " Image number " << count << " 2nd method " << optimalRadius << " " << optimum << std::endl;
      for(int j=0;j<points.size();j++) {
        std::cout << detections[i].widths[j] << std::endl;
      }
      std::cout << "----------------------------------------" << std::endl;
          myfile.close();
        }
  }//namespace objrec

  std::vector<float> ObjectDetector::calculateOptimum(std::vector<float> x, std::vector<float> y, int n, int N) {
    int i,j,k;
    float X[2*n+1];                        //Array that will store the values of sigma(xi),sigma(xi^2),sigma(xi^3)....sigma(xi^2n)
    for (i=0;i<2*n+1;i++)
    {
        X[i]=0;
        for (j=0;j<N;j++)
            X[i]=X[i]+pow(x[j],i);        //consecutive positions of the array will store N,sigma(xi),sigma(xi^2),sigma(xi^3)....sigma(xi^2n)
    }
    float B[n+1][n+2],a[n+1];            //B is the Normal matrix(augmented) that will store the equations, 'a' is for value of the final coefficients
    for (i=0;i<=n;i++)
        for (j=0;j<=n;j++)
            B[i][j]=X[i+j];            //Build the Normal matrix by storing the corresponding coefficients at the right positions except the last column of the matrix
    float Y[n+1];                    //Array to store the values of sigma(yi),sigma(xi*yi),sigma(xi^2*yi)...sigma(xi^n*yi)
    for (i=0;i<n+1;i++)
    {
        Y[i]=0;
        for (j=0;j<N;j++)
        Y[i]=Y[i]+pow(x[j],i)*y[j];        //consecutive positions will store sigma(yi),sigma(xi*yi),sigma(xi^2*yi)...sigma(xi^n*yi)
    }
    for (i=0;i<=n;i++)
        B[i][n+1]=Y[i];                //load the values of Y as the last column of B(Normal Matrix but augmented)
    n=n+1;                //n is made n+1 because the Gaussian Elimination part below was for n equations, but here n is the degree of polynomial and for n degree we get n+1 equations
    for (i=0;i<n;i++)                    //From now Gaussian Elimination starts(can be ignored) to solve the set of linear equations (Pivotisation)
        for (k=i+1;k<n;k++)
            if (B[i][i]<B[k][i])
                for (j=0;j<=n;j++)
                {
                    float temp=B[i][j];
                    B[i][j]=B[k][j];
                    B[k][j]=temp;
                }

    for (i=0;i<n-1;i++)            //loop to perform the gauss elimination
        for (k=i+1;k<n;k++)
            {
                float t=B[k][i]/B[i][i];
                for (j=0;j<=n;j++)
                    B[k][j]=B[k][j]-t*B[i][j];    //make the elements below the pivot elements equal to zero or elimnate the variables
            }
    for (i=n-1;i>=0;i--)                //back-substitution
    {                        //x is an array whose values correspond to the values of x,y,z..
        a[i]=B[i][n];                //make the variable to be calculated equal to the rhs of the last equation
        for (j=0;j<n;j++)
            if (j!=i)            //then subtract all the lhs values except the coefficient of the variable whose value                                   is being calculated
                a[i]=a[i]-B[i][j]*a[j];
        a[i]=a[i]/B[i][i];            //now finally divide the rhs by the coefficient of the variable to be calculated
    }
    std::vector<float> results;
    float optimum = -a[1]/(2*a[2]);
    results.push_back(optimum);
    results.push_back(a[2]*pow(optimum,2) + a[1]*pow(optimum,1) + a[0]);
    return results;
  }
}
}
