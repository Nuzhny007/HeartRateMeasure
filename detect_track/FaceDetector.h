#pragma once

#include "opencv2/opencv.hpp"

#ifdef USE_GPU
using namespace std;
#include "opencv2/gpu/gpu.hpp"
#endif

class FaceDetector
{
public:
    FaceDetector();
    ~FaceDetector();

    cv::Rect detect_biggest_face(cv::Mat& image);

private:
    double m_kw;
    double m_kh;

#ifdef USE_GPU
    cv::gpu::CascadeClassifier_GPU m_cascade;
#else
    cv::CascadeClassifier m_cascade;
#endif
};
