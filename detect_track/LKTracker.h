#pragma once
#include <opencv2/opencv.hpp>

///
/// \brief The LKTracker class
///
class LKTracker
{

public:
    LKTracker();
    LKTracker(cv::Rect initRegion);

    bool IsLost(void)
    {
        return lost;
    }

    cv::Rect GetTrackedRegion(void)
    {
        return bb2;
    }

    void Track(cv::Mat img);
    void ReinitTracker(cv::Rect initRegion);

 private:
    bool tvalid;
    bool tracked;

    std::vector<cv::Point2f> pointsFB;
    cv::Size window_size;
    int level;
    std::vector<uchar> status;
    std::vector<uchar> FB_status;
    std::vector<float> similarity;
    std::vector<float> FB_error;


    float simmed;
    float fbmed;
    cv::TermCriteria term_criteria;
    float lambda;
    void NormCrossCorrelation(const cv::Mat& img1,const cv::Mat& img2, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2);
    bool FilterPts(std::vector<cv::Point2f>& points1,std::vector<cv::Point2f>& points2);

    void BbPoints(std::vector<cv::Point2f>& points,const cv::Rect& bb);
    void BbPredict(const std::vector<cv::Point2f>& points1,const std::vector<cv::Point2f>& points2);
    bool Trackf2f(const cv::Mat& img1, const cv::Mat& img2, std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2);

    bool lost;
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    cv::Rect bb1;
    cv::Rect bb2;

    cv::Mat imgPrev;

    float getFB()
    {
        return fbmed;
    }
};
