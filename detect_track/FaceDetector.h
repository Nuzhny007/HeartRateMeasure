#pragma once

#include "opencv2/opencv.hpp"

class FaceDetector
{
public:
    FaceDetector();
    ~FaceDetector();

    cv::Rect detect_biggest_face(cv::Mat& image, bool originalFace);

private:
    double m_kw;
    double m_kh;

    cv::CascadeClassifier m_cascade;
};
