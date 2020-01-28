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

    bool IsLost(void) const;
    cv::Rect GetTrackedRegion(void) const;
    bool GetMovingSum(cv::Point2d& moveSum) const;

    void Track(cv::Mat img);
    void ReinitTracker(cv::Rect initRegion, const std::vector<cv::Point2f>& points);

	void GetPoints(std::vector<cv::Point2f>& points);

 private:
    bool m_tvalid = false;
    bool m_tracked = false;

    std::vector<cv::Point2f> m_pointsFB;
    cv::Size m_windowSize;
    int m_level = 5;
    std::vector<uchar> m_status;
    std::vector<uchar> m_FBStatus;
    std::vector<float> m_similarity;
    std::vector<float> m_FBError;


    float m_simmed = 0.0f;
    float m_fbmed = 0.0f;
    cv::TermCriteria m_termCriteria;
    float m_lambda = 0.005f;
    void NormCrossCorrelation(const cv::Mat& img1, const cv::Mat& img2);
    bool FilterPts();

    void BbPoints(std::vector<cv::Point2f>& points, const cv::Rect& bb);
    void BbPredict();
    bool Trackf2f(const cv::Mat& img1, const cv::Mat& img2);

    bool m_lost = false;
    std::vector<cv::Point2f> m_points1;
    std::vector<cv::Point2f> m_points2;
    cv::Rect m_bb1;
    cv::Rect m_bb2;

    cv::Mat m_imgPrev;

    static const size_t MinPoints = 20;
    static const size_t PointsToGenerate = 30;
};
