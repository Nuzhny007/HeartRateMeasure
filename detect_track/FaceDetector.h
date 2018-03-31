#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/face.hpp>

///
/// \brief The FaceDetectorBase class
///
class FaceDetectorBase
{
public:
    FaceDetectorBase(bool /*seOCL*/)
    {

    }

    virtual ~FaceDetectorBase()
    {

    }

    virtual cv::Rect DetectBiggestFace(cv::UMat image) = 0;
};

///
/// \brief The FaceDetectorHaar class
///
class FaceDetectorHaar : public FaceDetectorBase
{
public:
    FaceDetectorHaar(bool useOCL);
    ~FaceDetectorHaar();

    cv::Rect DetectBiggestFace(cv::UMat image);

private:
    double m_kw;
    double m_kh;

    cv::CascadeClassifier m_cascade;
};

///
/// \brief The FaceDetectorDNN class
///
class FaceDetectorDNN : public FaceDetectorBase
{
public:
    FaceDetectorDNN(bool useOCL);
    ~FaceDetectorDNN();

    cv::Rect DetectBiggestFace(cv::UMat image);

private:
    cv::String m_modelConfiguration;
    cv::String m_modelBinary;
    cv::dnn::Net m_net;

    float m_confidenceThreshold;
};

///
/// \brief The FaceLandmarksDetector class
///
class FaceLandmarksDetector
{
public:
    FaceLandmarksDetector();
    ~FaceLandmarksDetector();

    void Detect(cv::UMat image, const cv::Rect& faceRect, std::vector<cv::Point2f>& landmarks);

private:
    std::string m_modelFilename;
    cv::Ptr<cv::face::FacemarkKazemi> m_facemark;
};
