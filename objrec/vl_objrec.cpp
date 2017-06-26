#include "object_detection.h"

#include <fstream>
// points1 = model, points2 = test image
// A greedy algorithm finding the average distance of a model kp to the nearest object kp

namespace vislab
{
  namespace objrec
  {

bool sortpair(const std::pair<double, cv::KeyPoint>& lhs, const std::pair<double, cv::KeyPoint>& rhs) 
{ 
  return lhs.first < rhs.first; 
}
static bool sortfloatpair(const std::pair<float, cv::KeyPoint>& lhs, const std::pair<float, cv::KeyPoint>& rhs) 
{ 
  return lhs.first < rhs.first; 
}

// static bool sortpair2(const std::pair<double, unsigned int>& lhs, const std::pair<double, unsigned int>& rhs) 
// { 
//   return lhs.first < rhs.first; 
// }

double kpdistance( const std::vector<cv::KeyPoint> &points1, const std::vector<cv::KeyPoint> &points2, double maxdist )
{
    double result = 0;

    // We want the number of keypoints to be roughly the same, and punish large differences
    double ratio = (double) points1.size() / points2.size();
    if(ratio < 1) ratio = 1/ratio;

    std::vector< std::pair<double,cv::KeyPoint> > matches; 
    std::vector<double> sorted_distances;

    // Get the minimum distance for each element in points1
    for(unsigned i=0; i<points1.size(); i++)
    {
        double x1 = points1[i].pt.x, y1 = points1[i].pt.y;
        std::vector<double> distances; 

        for(unsigned j=0; j<points2.size(); j++)
        {
            double x2 = points2[j].pt.x, y2 = points2[j].pt.y;
            double curdist = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2);

            // Only consider points at the same scale
            double scaleratio = points1[i].size / points2[j].size;
            if(scaleratio > 0.91 && scaleratio < 1.1) distances.push_back(curdist);
        }
        std::sort(distances.begin(), distances.end());

        if(distances.size() > 0) matches.push_back(std::make_pair(distances[0],points1[i]));
    }

    std::sort(matches.begin(), matches.end(), sortpair);

    unsigned number2 = points2.size();
    for(unsigned i=0; i<matches.size(); i++)
    {
        double x1 = matches[i].second.pt.x, y1 = matches[i].second.pt.y;
        std::vector< std::pair<double,int> > distances; 

        if(points2.empty()) break;

        // double interimsum = 0;

        for(unsigned j=0; j<number2; j++)
        {
            double x2 = points2[j].pt.x, y2 = points2[j].pt.y;
            double curdist = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2);

            // interimsum += sqrt(curdist);
            // if((interimsum/(sorted_distances.size()+1) * ratio) > maxdist) return maxdist+1;

            distances.push_back(std::make_pair(curdist,j));
        }
        std::sort(distances.begin(),distances.end());

        // distances[0].first is the shortest distance, distances[0].second is the index of the nearest keypoint
        sorted_distances.push_back(distances[0].first);

        // Swap the nearest keypoint with the last one in the list and shorten the list
        // cv::KeyPoint temp;
        // temp = points2[points2.size()-1];
        // points2[points2.size()-1] = points2[distances[0].second];
        // points2[distances[0].second] = temp;
        // points2.pop_back();
    }

    // int cnt = 0;
    for(unsigned i=0; i<sorted_distances.size(); i++)
    {
        result += sqrt(sorted_distances[i]);
        // if(sqrt(sorted_distances[i]) < 5) 
        // {
        //     cnt++;
        //     result += sqrt(sorted_distances[i]);
        // }
    }
    
    result /= sorted_distances.size();
    result *= ratio;
    // result /= cnt;
    // result = result * ratio; 

    return result;
}

// double descdist(const std::vector<unsigned char> &first, const std::vector<unsigned char> &second);
// 
// double kpdescdistance( const std::vector<cv::KeyPoint> &points1,  const std::vector<cv::KeyPoint> &points2, 
//                        const std::vector<descriptor> &descs1, const std::vector<descriptor> &descs2 )
// {
//     double result = 0;
// 
//     // We want the number of keypoints to be roughly the same, and punish large differences
//     double ratio = (double) points1.size() / points2.size();
//     if(ratio < 1) ratio = 1/ratio;
// 
//     // int count = 0;
// 
//     // Get the minimum distance for each element in points1
//     for(unsigned i=0; i<points1.size(); i++)
//     {
//         double x1 = points1[i].pt.x, y1 = points1[i].pt.y;
//         // std::vector<double> distances; 
//         std::vector< std::pair<double,unsigned int> > matches; 
// 
//         for(unsigned j=0; j<points2.size(); j++)
//         {
//             double x2 = points2[j].pt.x, y2 = points2[j].pt.y;
//             double curdist = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2);
// 
//             // Only consider points at the same scale
//             double scaleratio = points1[i].size / points2[j].size;
//             if(scaleratio > 0.91 && scaleratio < 1.1) matches.push_back(std::make_pair(curdist,j));
//         }
//         std::sort(matches.begin(), matches.end());
// 
//         double smallestval=100000;
//         for(unsigned k=0; k<matches.size() && k<10; k++)
//         {
//             double val;
// 
//             val = descdist(descs1[i],descs2[matches[k].second]);
//             val = val * matches[k].first;
//             
//             // if(matches[k].first < 30 && val==0)
//             // {
//             //     count++;
//             //     break;
//             // }
// 
//             if(val<smallestval) smallestval = val;
//         }
//         result += smallestval;
//     }
//     
//     // result = count / (double) points1.size();
//     result *= ratio;
// 
//     return result;
// }
// 
// double descdist(const std::vector<unsigned char> &first, const std::vector<unsigned char> &second)
// {
//     if(first.size() != second.size()) return -1;
// 
//     double result=0;
// 
//     // // Euclidean distance
//     // for(unsigned i=0; i<first.size() && i<second.size(); i++)
//     // {
//     //     result += (first[i]-second[i])*(first[i]-second[i]);
//     // }
//     // return sqrt(result);
// 
//     // Hamming distance
//     for(unsigned i=0; i<first.size() && i<second.size(); i++)
//     {
//         if(first[i] != second[i]) result++;
//     }
//     return result;
// 
// }

// double descdist(const std::vector<unsigned char> &first, const std::vector<unsigned char> &second);

double kpdescdistance( const std::vector<cv::KeyPoint> &points1,  const std::vector<cv::KeyPoint> &points2, 
                       const std::vector<cv::Mat> &descs1, const std::vector<cv::Mat> &descs2 )
{
    double result = 0;

    // We want the number of keypoints to be roughly the same, and punish large differences
    double ratio = (double) points1.size() / points2.size();
    if(ratio < 1) ratio = 1/ratio;

    // int count = 0;

    cv::Mat desc1, desc2;
    // Get the minimum distance for each element in points1
    for(unsigned i=0; i<points1.size(); i++)
    {
        descs1[i].convertTo( desc1, CV_32F );

        double smallestval=100000;
        for(unsigned k=0; k<points2.size(); k++)
        {
            double val;
            descs2[k].convertTo( desc2, CV_32F );

            cv::Mat cor;
            int match_method=CV_TM_CCORR_NORMED;
            /// Create the result matrix
            int result_cols = desc1.cols - desc1.cols + 1;
            int result_rows = desc1.rows - desc1.rows + 1;

            cor.create( result_cols, result_rows, CV_32FC1 );

            /// Do the Matching and Normalize
            matchTemplate( desc1, desc2, cor, match_method );
            normalize( cor, cor, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );

            double min, max;
            minMaxLoc(cor, &min, &max);
            
            val = max; 

            if(val<smallestval) smallestval = val;
        }
        result += smallestval;
    }

    //     double x1 = points1[i].pt.x, y1 = points1[i].pt.y;
    //     // std::vector<double> distances; 
    //     std::vector< std::pair<double,unsigned int> > matches; 

    //     for(unsigned j=0; j<points2.size(); j++)
    //     {
    //         double x2 = points2[j].pt.x, y2 = points2[j].pt.y;
    //         double curdist = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2);

    //         // Only consider points at the same scale
    //         double scaleratio = points1[i].size / points2[j].size;
    //         if(scaleratio > 0.91 && scaleratio < 1.1) matches.push_back(std::make_pair(curdist,j));
    //     }
    //     std::sort(matches.begin(), matches.end());

    //     double smallestval=100000;
    //     for(unsigned k=0; k<matches.size() && k<10; k++)
    //     {
    //         double val;

    //         Mat cor;
    //         filter2D( descs1[i],cor,-1,descs2[matches[k].second]); 
    //         Scalar _sum = sum(cor);

    //         std::cout << sum(descs1[i])[0] << std::endl;

    //         val = 1/_sum[0] * matches[k].first;
    //         
    //         if(val<smallestval) smallestval = val;
    //     }
    //     result += smallestval;
    // }
    
    // result = count / (double) points1.size();
    result *= ratio;

    return result;
}

double kpsiftdescdistance( const std::vector<cv::KeyPoint> &points1,  const std::vector<cv::KeyPoint> &points2, 
                       const cv::Mat &descs1, const cv::Mat &descs2 )
{
    double result = 0;

    // We want the number of keypoints to be roughly the same, and punish large differences
    double ratio = (double) points1.size() / points2.size();
    if(ratio < 1) ratio = 1/ratio;

    // int count = 0;

    // std::cout << points1.size() << ", " << points2.size() << std::endl;
    // return result;
    
    std::vector<std::pair<float,cv::KeyPoint> > distances, distances2;

    // For each point in image1, find the most similar point in image2
    for(unsigned i=0; i<points1.size(); i++)
    {
        float minval=10000;
        cv::KeyPoint minkp;

        for(unsigned j=0; j<points2.size(); j++)
        {
            float val=0;
            // Mat d1 = descs1.row(i), d2 = descs2.row(j);
            const float *d1 = descs1.ptr<float>(i);
            const float *d2 = descs2.ptr<float>(j);
            for(int m=0; m<descs1.cols; m++)
                val+= (d1[m]-d2[m])*(d1[m]-d2[m]);
                // val+= (d1.at<float>(0,m)-d2.at<float>(0,m)) *(d1.at<float>(0,m)-d2.at<float>(0,m));
            val = sqrt(val);

            if(minval>val)  
            {
                minval = val; 
                minkp = points2[j];
            }
            
            // std::cout << val << " - " << minval << std::endl; 
        }
        // result+=minval;
        //distances.push_back(std::make_pair<float,cv::KeyPoint>(minval,points1[i]));
        //distances2.push_back(std::make_pair<float,cv::KeyPoint>(minval,minkp));
        distances.push_back(std::make_pair(minval,points1[i]));
        distances2.push_back(std::make_pair(minval,minkp));
            // std::cout << minval << std::endl;
    }

    // sort by similarity
    std::sort(distances.begin(),distances.end(), sortfloatpair);
    std::sort(distances2.begin(),distances2.end(), sortfloatpair);

    // For each pair of similar points, calculate the difference in position
    for(unsigned l=0; l<distances.size() && l<35; l++)
    {
        cv::KeyPoint pt1 = distances[l].second;
        cv::KeyPoint pt2 = distances2[l].second;
        float xdiff = pt1.pt.x-pt2.pt.x;
        float ydiff = pt1.pt.y-pt2.pt.y;
        float minkpdist = sqrt(xdiff*xdiff + ydiff*ydiff);

        double score = distances[l].first * minkpdist;
        result += score;
    }
    
    result /= 35;

    // float result2 = 0;
    // std::vector<float> distdiff;

    // for(unsigned m=0; m<distances.size()-1 && m<29; m++)
    // {
    //     cv::KeyPoint pt1a = distances[m].second;
    //     cv::KeyPoint pt2a = distances2[m].second;

    //     float mindist = 100000;
    //     for(unsigned n=m+1; n<distances.size() && n<30; n++)
    //     {
    //         cv::KeyPoint pt1b = distances[n].second;
    //         cv::KeyPoint pt2b = distances2[n].second;

    //         float xdiffa = (pt1a.pt.x-pt2a.pt.x)/pt1a.size;
    //         float ydiffa = (pt1a.pt.y-pt2a.pt.y)/pt1a.size;
    //         float xdiffb = (pt1b.pt.x-pt2b.pt.x)/pt1b.size;
    //         float ydiffb = (pt1b.pt.y-pt2b.pt.y)/pt1b.size;

    //         // float factor = ( sqrt((xdiffa*xdiffa)+(ydiffa*ydiffa))); //+sqrt((xdiffb*xdiffb)+(ydiffb*ydiffb)) )/2;

    //         float dd = sqrt((xdiffa-xdiffb)*(xdiffa-xdiffb) + (ydiffa-ydiffb)*(ydiffa-ydiffb));
    //         distdiff.push_back(dd);
    //     }
    // }

    // std::sort(distdiff.begin(), distdiff.end());
    // int cnt=0;
    // for(unsigned o=0; o<distdiff.size(); o++)
    // {
    //     result2 += distdiff[o];
    // }
    // result2 /= distdiff.size();

    // result = result2;

    // for(unsigned l=0; l<distances.size()-1 && l<9; l++)
    // {
    //     for(unsigned m=l+1; m<distances.size() && m<10; m++)
    //     {
    //         double score = 
    //     }
    // }

    // Get the minimum distance for each element in points1
    // for(unsigned i=0; i<points1.size(); i++)
    // {
    //     double x1 = points1[i].pt.x, y1 = points1[i].pt.y;
    //     // std::vector<double> distances; 
    //     std::vector< std::pair<double,unsigned int> > matches; 

    //     for(unsigned j=0; j<points2.size(); j++)
    //     {
    //         double x2 = points2[j].pt.x, y2 = points2[j].pt.y;
    //         double curdist = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2);

    //         // Only consider points at the same scale
    //         double scaleratio = points1[i].size / points2[j].size;
    //         if(scaleratio > 0.91 && scaleratio < 1.1) matches.push_back(std::make_pair(curdist,j));
    //     }
    //     std::sort(matches.begin(), matches.end());

    //     double smallestval=100000;
    //     for(unsigned k=0; k<matches.size() && k<10; k++)
    //     {
    //         double val;

    //         Mat d1 = descs1.row(1), d2 = descs2.row(matches[k].second);
    //         for(int m=0; m<d1.cols; m++)
    //             val+= (d1.at<float>(0,m)-d2.at<float>(0,m)) *(d1.at<float>(0,m)-d2.at<float>(0,m));
    //         val = sqrt(val);

    //         // val = descdist(descs1[i],descs2[matches[k].second]);

    //         // Mat cor;
    //         // filter2D( descs1[i],cor,-1,descs2[matches[k].second]); //,Point(-1,-1),0,BORDER_CONSTANT);
    //         // Scalar _sum = sum(cor);
    //         // val = 1/_sum[0] * matches[k].first;
    //         
    //         // if(matches[k].first < 30 && val==0)
    //         // {
    //         //     count++;
    //         //     break;
    //         // }

    //         if(val*matches[k].first<smallestval) smallestval = val*matches[k].first;
    //     }
    //     result += smallestval;
    // }
    
    // result = count / (double) points1.size();
    result *= ratio;

    return result;
}

double kpcolhistdistance( const std::vector<cv::KeyPoint> &points1,  const std::vector<cv::KeyPoint> &points2, 
                       const std::vector<cv::MatND> &descs1, const std::vector<cv::MatND> &descs2 )
{
    double result = 0;

    // We want the number of keypoints to be roughly the same, and punish large differences
    double ratio = (double) points1.size() / points2.size();
    if(ratio < 1) ratio = 1/ratio;
    
    int cnt = 0;

    // int count = 0;

    // Get the minimum distance for each element in points1
    for(unsigned i=0; i<points1.size(); i++)
    {
        double x1 = points1[i].pt.x, y1 = points1[i].pt.y;
        // std::vector<double> distances; 
        std::vector< std::pair<double,unsigned int> > matches; 

        for(unsigned j=0; j<points2.size(); j++)
        {
            double x2 = points2[j].pt.x, y2 = points2[j].pt.y;
            double curdist = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2);

            // Only consider points at the same scale
            double scaleratio = points1[i].size / points2[j].size;
            if(scaleratio > 0.91 && scaleratio < 1.1) matches.push_back(std::make_pair(curdist,j));
        }
        std::sort(matches.begin(), matches.end());

        double smallestval=1e7;
        for(unsigned k=0; k<matches.size() && k<10; k++)
        {
            // if(matches[k].first > 15) break;
            double val;

            // val = descdist(descs1[i],descs2[matches[k].second]);
            // double score = 
            compareHist(descs1[i], descs2[matches[k].second], CV_COMP_CORREL);

            val = matches[k].first;

            // std::cout << score << std::endl;

            // if(score > -100) // match
            {
                if(val<smallestval) smallestval = val;
            }
            
            // if(matches[k].first < 30 && val==0)
            // {
            //     count++;
            //     break;
            // }
            // std::cout << val << std::endl;

            // if(val<smallestval) smallestval = val;
        }
        // if(smallestval < 1e6) 
        {
            result += smallestval;
            cnt++;
        }
    }
    
    // result = count / (double) points1.size();
    result *= ratio;
    // result *= (float)cnt/points1.size();
    // if(points1.size() == 0) std::cout << "AAAAAAAAAAA" << std::endl;
    // std::cout << result << std::endl;

    return result;
}

double descdist(const std::vector<unsigned char> &first, const std::vector<unsigned char> &second)
{
    if(first.size() != second.size()) return -1;

    double result=0;

    // // Euclidean distance
    // for(unsigned i=0; i<first.size() && i<second.size(); i++)
    // {
    //     result += (first[i]-second[i])*(first[i]-second[i]);
    // }
    // return sqrt(result);

    // Hamming distance
    for(unsigned i=0; i<first.size() && i<second.size(); i++)
    {
        if(first[i] != second[i]) result++;
    }
    return result;

}

// puts a Gaussian on top of each keypoint, with sigma proportional to lambda
void makegauss( cv::Mat &out, std::vector<cv::KeyPoint> &points)
{
    for(unsigned i=0; i<points.size(); i++)
    {
        float xpos = points[i].pt.x;
        float ypos = points[i].pt.y;
        float lambda = points[i].size;

        for(int x=0; x<out.cols; x++)
        {
            for(int y=0; y<out.rows; y++)
            {
                float val = 50*exp(- (x-xpos)*(x-xpos)/(2*lambda*lambda) - (y-ypos)*(y-ypos)/(2*lambda*lambda) );
                out.at<float>(y,x) += val;
            }
        }
    }
    imwrite("temp.png", out);
}

int classifyObjectsSiftNNFast( vislab::evaluation::ImgInfo& files_test, std::vector<vislab::evaluation::ImgInfo>& classes_train, std::vector<cv::flann::Index*> finders, const unsigned numClasses, const int knnSearchParamsChecks )
{
    int bestlabelnum = -1;

    std::vector<double> votesfinal(numClasses,0);

    // Use coarse-to-fine strategy to speed up the processing
    // bool speedup = true;  
    bool speedup = false;

    // double t1 = (double)getTickCount();

    double curlambda = 128;
    bool finished = false;

    // std::cout << std::endl;
    while(!finished)
    {
        int ccl = 0;
        for(unsigned papa=0; papa<votesfinal.size();papa++)
            if(votesfinal[papa]>=0) ccl++;
        if(ccl == 1) break;

        for(int d=files_test.points.size()-1; d>=0; d--)
        {
            if(files_test.points[d].size > curlambda+1) continue;
            if(files_test.points[d].size < curlambda-1) break;
            // Here I have a testing keypoint

            // First test the coarse keypoints against all models. Later, progressively remove least similar classes

            for(unsigned i=0; i<classes_train.size(); i++)
            {   
	        cv::flann::Index *finder = finders[i];

                // if(votesfinal[classes_train[i].labelnum] < 0) continue;

                int neighbours = 1;
                std::vector<int>   knnin(neighbours);
                std::vector<float> knndis(neighbours);
                cv::Mat curdesc = files_test.descs3.row(d);
                std::vector<float> query;
                for(int foo=0; foo<curdesc.cols; foo++)
                    query.push_back(curdesc.at<float>(0,foo));
                finder->knnSearch( query, knnin, knndis, neighbours, cv::flann::SearchParams(knnSearchParamsChecks));

		//cv::KeyPoint& pt1 = files_test.points[d]; // warning: unused variable
		//cv::KeyPoint& pt2 = classes_train[i].points[knnin[0]]; // warning: unused variable

		// Only compare if within two octaves! Cuts the time significantly
		//double lambda1 = pt1.size; // warning: unused variable

		// Only compare if within two octaves! Cuts the time significantly
		//double lambda1 = pt1.size; // warning: unused variable
		//double lambda2 = pt2.size; // warning: unused variable
		// if( lambda1/lambda2 > 2.1 || lambda2/lambda1 > 2.1) continue;
		
		//float xdiff = pt1.pt.x-pt2.pt.x; // warning: unused variable
		//float ydiff = pt1.pt.y-pt2.pt.y; // warning: unused variable
		//float dist = sqrt(xdiff*xdiff + ydiff*ydiff); // warning: unused variable
		
		float val = knndis[0];
		
		val = val*val; // + 1*dist*dist;

                votesfinal[classes_train[i].labelnum] += val; //(si[0].first)*(si[0].first);
            }

        }

        // Eliminate all classes below a threshold, they slow us down and are unlikely to help
        if(speedup && curlambda < 64)
        {
            double max1 = 0;
            double min1 = 0;
            
            for(unsigned m=0; m<votesfinal.size(); m++)
                if(votesfinal[m] > max1)
                    max1 = votesfinal[m];

            min1 = max1;
            for(unsigned m=0; m<votesfinal.size(); m++)
                if(votesfinal[m] < min1 && votesfinal[m]>0) 
                    min1 = votesfinal[m];

            // std::cout << "Min1: " << min1 << std::endl;

            for(unsigned m=0; m<votesfinal.size(); m++)
                if(votesfinal[m] > max1-1)
                    votesfinal[m] = -1;

            int cntgone = 0;
            for(unsigned m=0; m<votesfinal.size(); m++)
            {
                if(votesfinal[m]<0) cntgone++;
                // std::cout << votesfinal[m] << "   ";
            }
            // std::cout << std::endl;
            // std::cout << cntgone << " classes gone" << std::endl;
        }

        curlambda /= sqrt(2);
        if(curlambda< 2) finished = true;
        // finished = true;
    }   // while(!finished)

    // Now go through all the votes and pick the class with lowest total distance
    double minscore = 1e30;
    for(unsigned m=0; m<votesfinal.size(); m++)
        if(votesfinal[m] < minscore && votesfinal[m]>0 )
        {
            minscore = votesfinal[m];
            bestlabelnum = m;
        }

    // double time1 = ((double) getTickCount()-t1)/getTickFrequency();
    // std::cout << " " << time1 << std::endl;
    // for(unsigned m=0; m<votesfinal.size(); m++)
    //     std::cout << (int) (votesfinal[m]*100) << " ";
    // std::cout << std::endl;

    return bestlabelnum;
}

int classifyObjectsSiftLocalNN( vislab::evaluation::ImgInfo& files_test, std::vector<int>& labels, cv::flann::Index* finder, const unsigned numClasses, const int knnSearchParamsChecks )
{
    int bestlabelnum = -1;

    std::vector<double> votesfinal(numClasses,0);

    // Use coarse-to-fine strategy to speed up the processing
    // bool speedup = true;  
    //bool speedup = false; // warning: unused variable

    // double t1 = (double)getTickCount();

    double curlambda = 128;
    bool finished = false;

    std::vector<int> numlabels(numClasses,0);
    for(unsigned h=0; h<labels.size(); h++)
    {   
      if(labels[h] >= static_cast<int>(numClasses)) continue;
        numlabels[labels[h]]++;
    }
    // for(unsigned h=0; h<numlabels.size(); h++)
    //     std::cout << numlabels[h] << " -- ";
    // std::cout << std::endl;

    // std::cout << std::endl;
    while(!finished)
    {
        for(int d=files_test.points.size()-1; d>=0; d--)
        {
            if(files_test.points[d].size > curlambda+1) continue;
            if(files_test.points[d].size < curlambda-1) break;
            // Here I have a testing keypoint

            // First test the coarse keypoints against all models. Later, progressively remove least similar classes

            // int neighbours = 61;
            unsigned int neighbours = 11;

            std::vector<int>   knnin(neighbours);
            std::vector<float> knndis(neighbours);
            cv::Mat curdesc = files_test.descs3.row(d);
            std::vector<float> query = cv::Mat_<float>(curdesc);
            // std::vector<float> query;
            // for(int foo=0; foo<curdesc.cols; foo++)
            //     query.push_back(curdesc.at<float>(0,foo));
            finder->knnSearch( query, knnin, knndis, neighbours, cv::flann::SearchParams(knnSearchParamsChecks));

            std::vector<float> vs(numClasses,0);

            float sigmasq = 30000.*30000.;
            float distb = knndis[neighbours-1];
            distb *= distb;
            float otherprob = exp(-distb/(2*sigmasq))/numlabels[labels[knnin[neighbours-1]]];

            std::vector<bool> classused(numClasses,false);

            for(unsigned int n=0; n<neighbours-1; n++)
            {
                int curlabel = labels[knnin[n]];
                // if(classused[curlabel]) continue;

                float val = knndis[n];
                val = val*val; // + 1*dist*dist;

                vs[curlabel] += exp(-val/(2*sigmasq)); 

                // votesfinal[curlabel] += val - distb;
                classused[curlabel] = true;
            }

            for(unsigned lab=0; lab<numClasses; lab++)
            {
                if(vs[lab] <= 0) continue;
                float value = vs[lab]/numlabels[lab];
                // std::cout << "value: " << value <<  " increment " << log(value/otherprob) << std::endl;
                votesfinal[lab] -= log(value/otherprob); // - distb; 
            }

        }

        curlambda /= sqrt(2);
        if(curlambda<2) finished = true;
        // finished = true;
    }   // while(!finished)

    // Now go through all the votes and pick the class with lowest total distance
    double minscore = 0;
    // double maxscore = 0;
    // for(unsigned m=0; m<votesfinal.size(); m++)
    //     std::cout << votesfinal[m] << std::endl;

    for(unsigned m=0; m<votesfinal.size(); m++)
    {
        if(votesfinal[m] < minscore) // && votesfinal[m] > -1)
        // if(votesfinal[m] > maxscore && votesfinal[m]>0 )
        {
            minscore = votesfinal[m];
            // maxscore = votesfinal[m];
            bestlabelnum = m;
        }
    }
    // std::cout << votesfinal[0] << " -- " << minscore << std::endl;

    // std::cout << " ---- " << std::endl;

    return bestlabelnum;
}

// Calculate L1 distance between
// std::vector<float> newdistances( std::vector<float> &knndis, std::vector<int> &knnin, cv::Mat &descs, int number)
// {
//     std::vector<float> result;
// 
// 
//     return result;
// }

// For compatibility reasons
int classifyObjectsGenericLocalNN( vislab::evaluation::ImgInfo& files_test, std::vector<int>& labels, cv::flann::Index* finder, const unsigned numClasses, const int knnSearchParamsChecks )
{
    return classifyObjectsGenericLocalNN( files_test.descs3, labels, finder, numClasses, knnSearchParamsChecks);
}

int classifyObjectsGenericLocalNN( cv::Mat& descs, std::vector<int>& labels, cv::flann::Index* finder, const unsigned numClasses, const int knnSearchParamsChecks )
{
    std::vector<float> foo;
    return classifyObjectsGenericLocalNN( descs, labels, finder, numClasses, knnSearchParamsChecks, foo );
}

int classifyObjectsGenericLocalNN( cv::Mat& descs, std::vector<int>& labels, cv::flann::Index* finder, const unsigned numClasses, const int knnSearchParamsChecks, std::vector<float>& probs )
{
    int bestlabelnum = -1;

    float totalbdist = 0;

    std::vector<double> votesfinal(numClasses,0);

    // double t1 = (double)getTickCount();

    std::vector<int> numlabels(numClasses,0);
    for(unsigned h=0; h<labels.size(); h++)
    {   
        if(labels[h] >= static_cast<int>(numClasses)) continue;
        numlabels[labels[h]]++;
    }

    if(descs.rows == 0) std::cout << "Zero input!" << std::endl;
    for(int d=descs.rows-1; d>=0; d--)
    {
        // int neighbours = 61;
        unsigned int neighbours = 11;

        std::vector<int>   knnin(neighbours);
        std::vector<float> knndis(neighbours);
        cv::Mat curdesc = descs.row(d);
        
        if(descs.type() == 5)     // floating point type, floating point descriptor
        {
            std::vector<float> query = cv::Mat_<float>(curdesc);
            finder->knnSearch( query, knnin, knndis, neighbours, cv::flann::SearchParams(knnSearchParamsChecks));
        }
        else                                  // unsigned char type, binary descriptor
        {
            std::vector<int> intknndis(neighbours);
            std::vector<unsigned char> query = cv::Mat_<unsigned char>(curdesc);
            finder->knnSearch( query, knnin, intknndis, neighbours, cv::flann::SearchParams(knnSearchParamsChecks));
            for(unsigned foo=0; foo<intknndis.size(); foo++) knndis[foo] = intknndis[foo];
        }

        std::vector<float> vs(numClasses,0);

        // float sigmasq = 30000.*30000.;
        float distb = knndis[neighbours-1];
        // std::cout << distb << std::endl;
        distb *= distb;
        totalbdist += distb;
        // float otherprob = exp(-distb/(2*sigmasq))/numlabels[labels[knnin[neighbours-1]]];

        std::vector<bool> classused(numClasses,false);

        for(unsigned n=0; n<neighbours-1; n++)
        {
            // std::cout << knndis[n] << std::endl;
            if(knndis[n]*knndis[n] > distb) break;

            int curlabel = labels[knnin[n]];
            if(classused[curlabel]) continue;

            float val = knndis[n];
            // std::cout << val << std::endl;
            // if(val>0.25) break;
            val = val*val; // + 1*dist*dist;

            // vs[curlabel] += exp(-val); 

            votesfinal[curlabel] += val - distb;
            // votesfinal[curlabel] += sqrt(val) - sqrt(distb);
            classused[curlabel] = true;
        }
        // float sum=0;
        // for(unsigned curlabel=0; curlabel<vs.size(); curlabel++)
        // {
        //     if(vs[curlabel] == 0) vs[curlabel] = exp(-distb);
        //     sum += vs[curlabel];
        // }
        // for(unsigned curlabel=0; curlabel<vs.size(); curlabel++)
        // {
        //     vs[curlabel]/=sum;
        //     votesfinal[curlabel] += log(vs[curlabel]);
        //     // std::cout << vs[curlabel] << " ";
        // }
        // std::cout << std::endl;

    }

    // Now go through all the votes and pick the class with lowest total distance
    double minscore = 0;
    double maxscore = -1e10;

    for(unsigned m=0; m<votesfinal.size(); m++)
    {
        if(votesfinal[m] < minscore) // && votesfinal[m] > -1)
        {
            minscore = votesfinal[m];
            bestlabelnum = m;
        }
        if(votesfinal[m] > maxscore) // && votesfinal[m] > -1)
        {
            maxscore = votesfinal[m];
        }
    }
    // Normalise
    std::vector<double> myprobs(votesfinal.size());
    probs = std::vector<float>(myprobs.size(),0);
    float probsum = 0;
    // float tempsum = 0;
    // 
    probs[bestlabelnum] = 1; 

    for(unsigned m=0; m<votesfinal.size(); m++)
    {
         myprobs[m] = 1- ((votesfinal[m]-minscore) / (maxscore-minscore));
         probsum += myprobs[m];
    }
    for(unsigned m=0; m<myprobs.size(); m++)
    {
        probs[m] = myprobs[m]/probsum;
    }
    
    // for(unsigned m=0; m<votesfinal.size(); m++)
    // {
    //      // myprobs[m] = votesfinal[m] + totalbdist;
    //      // std::cout << votesfinal[m] << "  ";
    //      // myprobs[m] = exp(-myprobs[m]);
    //      myprobs[m] = exp(-votesfinal[m]);
    //      probsum += myprobs[m];
    //      // std::cout << votesfinal[m] + totalbdist << " -> " << myprobs[m] << std::endl;
    // }
    // for(unsigned m=0; m<myprobs.size(); m++)
    // {
    //     probs[m] = myprobs[m]/probsum;
    //     // tempsum+=probs[m];
    //     // std::cout << probs[m] << " ";
    // }
    // std::cout << std::endl;

    return bestlabelnum;
}

int classifyObjectsSegmentLocalNN( vislab::evaluation::ImgInfo& files_test, std::vector<int>& labels, cv::flann::Index* finder, unsigned numClasses, const int knnSearchParamsChecks )
{
    int bestlabelnum = -1;

    std::vector<double> votesfinal(numClasses,0);

    // double t1 = (double)getTickCount();

    std::vector<int> numlabels(numClasses,0);
    for(unsigned h=0; h<labels.size(); h++)
    {   
      if(labels[h] >= static_cast<int>(numClasses)) continue;
        numlabels[labels[h]]++;
    }

    for(int d=files_test.descs3.rows-1; d>=0; d--)
    {
        unsigned int neighbours = 11;

        std::vector<int>   knnin(neighbours);
        std::vector<float> knndis(neighbours);
        cv::Mat curdesc = files_test.descs3.row(d);
        std::vector<float> query = cv::Mat_<float>(curdesc);
        // for(int foo=0; foo<curdesc.cols; foo++)
        //     query.push_back(curdesc.at<float>(0,foo));
        finder->knnSearch( query, knnin, knndis, neighbours, cv::flann::SearchParams(knnSearchParamsChecks));

        std::vector<float> vs(numClasses,0);

        // std::cout << "Closest: ";
        // for(unsigned k=0; k<knnin.size(); k++)
        //     std::cout << knnin[k] << " ";
        // std::cout << std::endl;

        float distb = knndis[neighbours-1];
        // distb *= distb;
        // float sigmasq = 30000.*30000.;
        // float otherprob = exp(-distb/(2*sigmasq))/numlabels[labels[knnin[neighbours-1]]];

        std::vector<bool> classused(numClasses,false);

        for(unsigned n=0; n<neighbours-1; n++)
        {
            int curlabel = labels[knnin[n]];
            // std::cout << curlabel << " ";
            // if(classused[curlabel]) continue;

            float val = knndis[n];

            // val = val*val; // + 1*dist*dist;
            // vs[curlabel] += exp(-val/(2*sigmasq)); 

            votesfinal[curlabel] += val - distb;
            classused[curlabel] = true;
        }
        // std::cout << std::endl;

        // for(unsigned lab=0; lab<numClasses; lab++)
        // {
        //     if(vs[lab] <= 0) continue;
        //     float value = vs[lab]/numlabels[lab];
        //     // std::cout << "value: " << value <<  " increment " << log(value/otherprob) << std::endl;
        //     votesfinal[lab] -= log(value/otherprob); // - distb; 
        // }
    }

    // Now go through all the votes and pick the class with lowest total distance
    double minscore = 0;
    //double maxscore = 0; // warning: unused variable
    // for(unsigned m=0; m<votesfinal.size(); m++)
    //     std::cout << votesfinal[m] << std::endl;

    for(unsigned m=0; m<votesfinal.size(); m++)
    {
        if(votesfinal[m] < minscore) // && votesfinal[m] > -1)
        // if(votesfinal[m] > maxscore && votesfinal[m]>0 )
        {
            minscore = votesfinal[m];
            // maxscore = votesfinal[m];
            bestlabelnum = m;
        }
    }
    return bestlabelnum;
}


int classifyObjectsSiftLocalNNSoft( vislab::evaluation::ImgInfo& files_test, std::vector<std::vector<float> >& classprobs, cv::flann::Index* finder, const unsigned numClasses, const int knnSearchParamsChecks )
{
    int bestlabelnum = -1;

    std::vector<double> votesfinal(numClasses,0);

    // double t1 = (double)getTickCount();

    for(int d=files_test.points.size()-1; d>=0; d--)
    {
        // Here I have a testing keypoint

        unsigned int neighbours = 3;

        std::vector<int>   knnin(neighbours);
        std::vector<float> knndis(neighbours);
        cv::Mat curdesc = files_test.descs3.row(d);
        std::vector<float> query;
        for(int foo=0; foo<curdesc.cols; foo++)
            query.push_back(curdesc.at<float>(0,foo));
        finder->knnSearch( query, knnin, knndis, neighbours, cv::flann::SearchParams(knnSearchParamsChecks));


        float sigmasq = 10000.*10000.;
        // float distb = knndis[neighbours-1];
        // distb *= distb;
        // float otherprob = exp(-distb/(2*sigmasq)); ///numlabels[labels[knnin[neighbours-1]]];
        // float otherprob = exp(-distb/(2*sigmasq))/numlabels[labels[knnin[neighbours-1]]];

        std::vector<bool> classused(numClasses,false);
        std::vector<double> vs(numClasses,0);
        std::vector<double> vs_other(numClasses,0);


        for(unsigned n=0; n<neighbours; n++)
        {
            float val = knndis[n];
            val = val*val; // + 1*dist*dist;

            // if(classprobs.size() <= knnin[n])
                // std::cout << "A: " << classprobs.size() << ", " << knnin[n] << std::endl;

            for(unsigned curcl=0; curcl<numClasses; curcl++)
            {
                // if(classprobs[knnin[n]].size() <= curcl)
                    // std::cout << "B:   "<< classprobs[knnin[n]].size() << ", " << curcl << std::endl;

                float prob = classprobs[knnin[n]][curcl];
                // float prob = 1;
                // if(classused[curcl]) continue;
                // vs[curcl] += val; ///classprobs[knnin[n]][curcl]; 
                // if(prob > 0)
                // {
                //     vs[curcl] += (val - distb) * prob;
                //     classused[curcl] = true;
                // }
                vs[curcl] += exp(-val/(2*sigmasq)) * prob; 
                // std::cout << vs[curcl] << " ";
            }

        }

        for(unsigned lab=0; lab<numClasses; lab++)
            if(vs[lab] != 0)
                votesfinal[lab] += log(vs[lab]); ///otherprob);
            // votesfinal[lab] += vs[lab];

        // {
        //     if(vs[lab] <= 0) continue;
        //     float value = vs[lab]/numlabels[lab];
        //     // std::cout << "value: " << value <<  " increment " << log(value/otherprob) << std::endl;
        //     votesfinal[lab] -= log(value/otherprob); // - distb; 
        // }
    }

    // Now go through all the votes and pick the class with lowest total distance
    double minscore = 0;
    //double maxscore = 0; //removed warning: unused variable
    // for(unsigned m=0; m<votesfinal.size(); m++)
    //     std::cout << votesfinal[m] << std::endl;

    for(unsigned m=0; m<votesfinal.size(); m++)
    {
        if(votesfinal[m] < minscore) // && votesfinal[m] > -1)
        // if(votesfinal[m] > maxscore && votesfinal[m]>0 )
        {
            minscore = votesfinal[m];
            // maxscore = votesfinal[m];
            bestlabelnum = m;
        }
    }
    if(bestlabelnum < 0 || bestlabelnum >= static_cast<int>(numClasses)) std::cout << "OMG!" << std::endl;
    // std::cout << votesfinal[0] << " -- " << minscore << std::endl;

    // std::cout << " ---- " << std::endl;

    return bestlabelnum;
}


int detectObjectsSift( vislab::evaluation::ImgInfo& files_test, std::vector<vislab::evaluation::ImgInfo>& files_train, const unsigned numClasses )
{
    // double maxscore = -1e30;
    int bestlabelnum = -1;

    std::vector<double> votesfinal(numClasses,0);

    std::vector<cv::Mat> objCentre;
    for(unsigned tmpi=0; tmpi<numClasses; tmpi++)
        objCentre.push_back(cv::Mat::zeros(files_test.height,files_test.width,CV_32F));

    // Use coarse-to-fine strategy to speed up the processing
    bool speedup = true;  
    // bool speedup = false;

    // double t1 = (double)getTickCount();

    double curlambda = 64;
    bool finished = false;

    // std::cout << std::endl;
    while(!finished)
    {
        // int ccl = 0;
        // for(unsigned papa=0; papa<votesfinal.size();papa++)
        //     if(votesfinal[papa]>=0) ccl++;
        // if(ccl == 1) break;

        // Count how many kps at this scale
        int cntkps = 0;
        for(int d=files_test.points.size()-1; d>=0; d--)
        {
            if(files_test.points[d].size < curlambda+1 && files_test.points[d].size > curlambda-1) cntkps++;
            if(files_test.points[d].size < curlambda-1) break;
        }

        for(int d=files_test.points.size()-1; d>=0; d--)
        {
            if(files_test.points[d].size > curlambda+1) continue;
            if(files_test.points[d].size < curlambda-1) break;
            // Here I have a testing keypoint

            // First test the coarse keypoints against all models. Later, progressively remove least similar classes

            std::vector<std::pair<float,int> > si;  // sorted indices for indirect addressing
            std::vector<int> bestcl, centerx, centery;
            std::vector<float> distances;

            for(unsigned i=0; i<files_train.size(); i++)
            {   
                if(votesfinal[files_train[i].labelnum] < 0) continue;

                for(int e=files_train[i].points.size()-1; e>=0; e--)
                {
                    // Here I have a training keypoint

                    // Only compare if within two octaves! Cuts the time significantly
                    double lambda1 = files_test.points[d].size;
                    double lambda2 = files_train[i].points[e].size;
                    if( lambda1/lambda2 > 2.1 || lambda2/lambda1 > 2.1) continue;

                    cv::KeyPoint& pt1 = files_test.points[d];
                    cv::KeyPoint& pt2 = files_train[i].points[e];
                    float xdiff = pt1.pt.x-pt2.pt.x;
                    float ydiff = pt1.pt.y-pt2.pt.y;
                    float dist = sqrt(xdiff*xdiff + ydiff*ydiff);

                    float val = 0;
                    const float *d1 = files_test.descs3.ptr<float>(d);
                    const float *d2 = files_train[i].descs3.ptr<float>(e);
                    for(int m=0; m<files_test.descs3.cols; m++)
                        val+= (d1[m]-d2[m])*(d1[m]-d2[m]);

                    val = sqrt(val); // * dist;
                    //si.push_back(std::make_pair<float,int>(val, si.size()));
                    si.push_back(std::make_pair(val, si.size()));
                    bestcl.push_back(files_train[i].labelnum);
                    int offsetx = pt1.pt.x + files_train[i].width /2-pt2.pt.x; 
                    int offsety = pt1.pt.y + files_train[i].height/2-pt2.pt.y;
                    centerx.push_back(offsetx);
                    centery.push_back(offsety);
                    distances.push_back(dist);
                }
            }
            std::sort(si.begin(), si.end());

            // Here this point votes for his preferred class
            unsigned numvotes = numClasses/2;
            numvotes = 10;
            // std::cout << si.size() << std::endl;
            
            // Here this point votes for his preferred class
            if(si.size() < numvotes) numvotes= si.size();
            for(unsigned ja=0; ja<numvotes; ja++)
            {
                int index = si[ja].second;
                if(centery[index] >= 0 && centery[index]<objCentre[0].rows && 
                        centerx[index]>=0 && centerx[index]<objCentre[0].cols)
                    objCentre[bestcl[index]].at<float>(centery[index],centerx[index]) += (numvotes-ja);
            }
        }
    
        // Eliminate all classes below a threshold, they slow us down and are unlikely to help
        if(speedup && curlambda < 64)
        {
            std::vector<float> sums(objCentre.size(),0);
            float max1 = -1;

            for(unsigned m=0; m<objCentre.size(); m++)
            {
                sums[m] = sum(objCentre[m])[0];
                if(sums[m] > max1) max1 = sums[m];
            }

            for(unsigned m=0; m<objCentre.size(); m++)
                if(sums[m] < max1*0.3)
                    votesfinal[m] = -1;
        }

        curlambda /= sqrt(2);
        if(curlambda< 7) finished = true;
    }   // while(!finished)

    // Blur all the class-specific votes to get object centre and select the strongest object
    float maxall = 0;
    bestlabelnum = -1;
    for(unsigned m=0; m<objCentre.size(); m++)
    {
        if(votesfinal[m] < 0) continue;

        cv::Mat objc = objCentre[m];
        cv::Mat dst;

        // Filter with a Gaussian and find maximum
        GaussianBlur(objc, dst, cv::Size(), 40);
        double min, max;
        minMaxLoc(dst, &min, &max);

        // double max = sum(objc)[0];

        if(max>maxall)
        {
            maxall = max;
            bestlabelnum = m;
        }
    }

    // double time1 = ((double) getTickCount()-t1)/getTickFrequency();
    // std::cout << " " << time1 << std::endl;
    // for(unsigned m=0; m<votesfinal.size(); m++)
    //     std::cout << (int) (votesfinal[m]*100) << " ";
    // std::cout << std::endl;

    return bestlabelnum;
}

int detectObjectsSift_old( vislab::evaluation::ImgInfo& files_test, std::vector<vislab::evaluation::ImgInfo>& files_train, const unsigned numClasses )
{
    int bestlabelnum = -1;

    std::vector<cv::Mat> objCentre;
    for(unsigned tmpi=0; tmpi<numClasses; tmpi++)
        objCentre.push_back(cv::Mat::zeros(files_test.height,files_test.width,CV_32F));

    for(int d=files_test.points.size()-1; d>=0; d--)
    {
        // Here I have a testing keypoint

        std::vector<std::pair<float,int> > si;  // sorted indices for indirect addressing
        std::vector<int> bestcl, centerx, centery;
        std::vector<float> ratios, distances;

        for(unsigned i=0; i<files_train.size(); i++)
        {   
            float ratio = (float) files_test.points.size() / files_train[i].points.size();
            if(ratio > 1) ratio = 1/ratio;

            for(int e=files_train[i].points.size()-1; e>=0; e--)
            {
                // Here I have a training keypoint

                // Only compare if within two octaves! Cuts the time significantly
                double lambda1 = files_test.points[d].size;
                double lambda2 = files_train[i].points[e].size;
                if( lambda1/lambda2 > 2.1 || lambda2/lambda1 > 2.1) continue;

                cv::KeyPoint& pt1 = files_test.points[d];
                cv::KeyPoint& pt2 = files_train[i].points[e];
                float xdiff = pt1.pt.x-pt2.pt.x;
                float ydiff = pt1.pt.y-pt2.pt.y;
                float dist = sqrt(xdiff*xdiff + ydiff*ydiff);

                float val = 0;
                const float *d1 = files_test.descs3.ptr<float>(d);
                const float *d2 = files_train[i].descs3.ptr<float>(e);
                for(int m=0; m<files_test.descs3.cols; m++)
                    val+= (d1[m]-d2[m])*(d1[m]-d2[m]);

                val = sqrt(val); // * dist;
                //si.push_back(std::make_pair<float,int>(val, si.size()));
                si.push_back(std::make_pair(val, si.size()));
                bestcl.push_back(files_train[i].labelnum);
                int offsetx = pt1.pt.x + files_train[i].width /2-pt2.pt.x; 
                int offsety = pt1.pt.y + files_train[i].height/2-pt2.pt.y;
                // offsetx *= lambda1/lambda2;
                // offsety *= lambda1/lambda2;
                centerx.push_back(offsetx);
                centery.push_back(offsety);
                ratios.push_back(ratio);
                distances.push_back(dist);
                // distances.push_back(val);
            }
        }
        std::sort(si.begin(), si.end());

        // Here this point votes for his preferred class
        if(bestcl.size() < 30 || ratios.size() < 30 || si.size() < 30) continue;
        for(int ja=0; ja<30; ja++)
        {
            int index = si[ja].second;
            if(centery[index] >= 0 && centery[index]<objCentre[0].rows && 
                    centerx[index]>=0 && centerx[index]<objCentre[0].cols)
                objCentre[bestcl[index]].at<float>(centery[index],centerx[index]) += (10-ja);
        }
    }

    // Blur all the class-specific votes to get object centre and select the strongest object
    float maxall = 0;
    bestlabelnum = -1;
    for(unsigned m=0; m<objCentre.size(); m++)
    {
        cv::Mat objc = objCentre[m];
        cv::Mat dst;

        // Filter with a Gaussian
        GaussianBlur(objc, dst, cv::Size(), 30);

        // Find maximum
        double min, max;
        minMaxLoc(dst, &min, &max);

        if(max>maxall)
        {
            maxall = max;
            bestlabelnum = m;
        }
    }

    return bestlabelnum;
}

int detectObjectsSurfLocalNN( vislab::evaluation::ImgInfo& files_test, std::vector<int>& labels, cv::flann::Index* finder, std::vector<cv::KeyPoint> allpoints, std::vector<cv::Point2f> offsets, const unsigned numClasses, const int knnSearchParamsChecks )
{
    int bestlabelnum = -1;

    std::vector<double> votesfinal(numClasses,0);

    // double t1 = (double)getTickCount();

    std::vector<int> numlabels(numClasses,0);
    for(unsigned h=0; h<labels.size(); h++)
    {   
        if(labels[h] >= static_cast<int>(numClasses)) continue;
        numlabels[labels[h]]++;
    }
    
    std::vector<cv::Mat> objCentre;
    for(unsigned tmpi=0; tmpi<numClasses; tmpi++)
        objCentre.push_back(cv::Mat::zeros(files_test.height,files_test.width,CV_32F));


    for(int d=files_test.points.size()-1; d>=0; d--)
    {
        // int neighbours = 61;
        unsigned int neighbours = 11;

        std::vector<int>   knnin(neighbours);
        std::vector<float> knndis(neighbours);
        cv::Mat curdesc = files_test.descs3.row(d);
        std::vector<float> query = cv::Mat_<float>(curdesc);
        finder->knnSearch( query, knnin, knndis, neighbours, cv::flann::SearchParams(knnSearchParamsChecks));

        int curx = files_test.points[d].pt.x;
        int cury = files_test.points[d].pt.y;

        std::vector<float> vs(numClasses,0);

        float distb = knndis[neighbours-1];
        distb *= distb;

        std::vector<bool> classused(numClasses,false);

        for(unsigned n=0; n<neighbours-1; n++)
        {
            int curlabel = labels[knnin[n]];
            if(classused[curlabel]) continue;

            float val = knndis[n];
            val = val*val; // + 1*dist*dist;

            votesfinal[curlabel] += val - distb;
            classused[curlabel] = true;

            // offsetx and offsety
            // int nnx = allpoints[knnin[n]].pt.x;
            // int nny = allpoints[knnin[n]].pt.y;

            cv::Point2f offset = offsets[knnin[n]];
            int votex = curx + offset.x;
            int votey = cury + offset.y;

            if(votex >= 0 && votex < objCentre[curlabel].cols &&
               votey >= 0 && votey < objCentre[curlabel].rows)
            {
                objCentre[curlabel].at<float>(votey,votex) += val-distb;
            }
        }
    }

    // Now go through all the votes and pick the class with lowest total distance
    double minscore = 0;

    for(unsigned m=0; m<objCentre.size(); m++)
    {
        cv::Mat objc = objCentre[m];
        cv::Mat dst;

        // Filter with a Gaussian and find maximum
        cv::GaussianBlur(objc, dst, cv::Size(), 40);
        double min, max;
        cv::minMaxLoc(dst, &min, &max);
        
        if(min < minscore) // && votesfinal[m] > -1)
        {
            minscore = min;
            bestlabelnum = m;
        }
    }

    return bestlabelnum;
}

void drawdetections(cv::Mat &img, std::vector<Detection>& dets, std::vector<std::string>& objects, int width)
{
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
        // std::vector<std::vector<Point> > polys;
        // polys.push_back(dets[r].hull);
        // polylines(img, polys, true, colour); 

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
void drawgt(cv::Mat &img, std::vector<cv::Rect>& dets, int width )
{
    for(unsigned r=0; r<dets.size(); r++)
    {
        cv::Scalar colour = cv::Scalar(255,64,64);

        // Bounding Box
        cv::rectangle(img, dets[r], cv::Scalar(255,255,255),2*width);
        cv::rectangle(img, dets[r], colour,width);
        cv::circle( img, dets[r].tl()+cv::Point(dets[r].width/2,dets[r].height/2), 4, colour, -1 );
    }
}

void evaluatedets( vislab::evaluation::ImgInfo& files_info, std::vector<Detection>& dets, int* truepos, int *falsepos, float thresh, int labelnum )
{
    int _tp = 0;
    int _fp = 0;

    std::vector<cv::Rect> rois = files_info.ROI;
    std::vector<bool> used(rois.size() ,false);
    for(unsigned i=0; i<dets.size(); i++)
    {
        if(dets[i].strength < thresh) continue;
        if(dets[i].labelnum != labelnum) continue;

        bool found = false;
        for(unsigned j=0; j<rois.size(); j++)
        {
            cv::Rect _interbox = dets[i].bbox & rois[j];
            float _inter = _interbox.area();
            float _union = dets[i].bbox.area() + rois[j].area() - _inter;

            if(_inter/_union > 0.5 && dets[i].labelnum == files_info.objlabelnums[j] 
                                   && dets[i].labelnum == labelnum && !used[j]) 
            {
                found = true;
                used[j] = true;
            }
        }

        if(found) 
        {
            _tp++;
            dets[i].correct = true;
        }
        else 
        {
            _fp++;
            dets[i].correct = false;
        }
    }
    *truepos = _tp;
    *falsepos = _fp;
}

void evaluatealldets( std::vector<vislab::evaluation::ImgInfo>& files_info, std::vector<std::vector<Detection> >& alldets, int labelnum )
{
    if(files_info.size() != alldets.size())
    {
        std::cout << "ERROR. Need the same number of images!" << std::endl;
        return;
    }

    std::vector<float> thresholds;
    // Find all the detection strengths and sort them, to find the right threshold
    for(unsigned i=0; i<alldets.size(); i++)
        for(unsigned j=0; j<alldets[i].size(); j++)
            if(alldets[i][j].labelnum == labelnum)
                thresholds.push_back(alldets[i][j].strength);
    
    std::sort(thresholds.begin(), thresholds.end());

    float avgprecision = 0;
    float oldrecall = 1;
    float maxfscore = 0;
    float bestrecall = 0;
    float bestprecision = 0;
    float bestfp = 0;

    std::stringstream filename;
    filename << "results-" << labelnum << ".csv";
    std::ofstream resultsfile(filename.str().c_str());
    resultsfile << "Class: " << labelnum << ", " << files_info.size() << " images. "<< std::endl << std::endl;
    resultsfile << "Threshold, TP, FP, FN, Precision, Recall, FPPI, F-Score" << std::endl;

    for(unsigned t=0; t<thresholds.size(); t++)
    {
        float thresh = thresholds[t];

        int allanno, alldet, alltruepos, allfalsepos, correctdets;
        allanno = alldet = alltruepos = allfalsepos = correctdets = 0;

        for(unsigned j=0; j<files_info.size(); j++)
        {
            int truepos = 0, falsepos = 0; 
            evaluatedets( files_info[j], alldets[j], &truepos, &falsepos, thresh, labelnum );

            for(unsigned i=0; i<files_info[j].ROI.size(); i++)
                if( files_info[j].objlabelnums[i] == labelnum ) allanno++;

            for(unsigned i=0; i<alldets[j].size(); i++)
                if(alldets[j][i].strength >= thresh && alldets[j][i].labelnum == labelnum) alldet++;

            alltruepos += truepos;
            allfalsepos += falsepos;
        }

        if(alldet == 0) break;

        float recall    = float(alltruepos)/float(allanno);
        float precision = float(alltruepos)/float(alldet);
        float fppi      = float(allfalsepos)/float(files_info.size());
        float fscore    = 2 * precision * recall / (precision+recall);

        // if(oldrecall == 0) oldrecall = recall;

        float deltarecall = oldrecall-recall;
        avgprecision += precision * deltarecall;

        oldrecall = recall;

        if(fscore > maxfscore)
        {
            maxfscore = fscore;
            bestrecall = recall;
            bestprecision = precision;
            bestfp = float(allfalsepos)/float(alldet);
        }

        // if(fppi < 2)
        resultsfile << thresh << ", " << alltruepos << ", " << allfalsepos << ", " << allanno-alltruepos << ","
                    << precision << ", " << recall << ", " << fppi << ", " << fscore << std::endl;

        if(fppi == 0 ) break;
        // std::cout << allanno << " annotated objects in " << files_info.size() << " images." << std::endl;
    }
    
    resultsfile << std::endl;
    resultsfile << "Average Precision: " << avgprecision << std::endl << std::endl;
    resultsfile << "Best value at, F-Score, " << maxfscore << std::endl;
    resultsfile << ", Recall, " << bestrecall << std::endl;
    resultsfile << ", Precision, " << bestprecision << std::endl;
    resultsfile << ", FP rate, " << bestfp << std::endl;
    resultsfile.close();


}


  } //namespace objrec
}//namespace vislab
