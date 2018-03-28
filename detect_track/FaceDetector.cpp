#include "FaceDetector.h"

///
/// \brief FaceDetectorHaar::FaceDetectorHaar
///
FaceDetectorHaar::FaceDetectorHaar(bool useOCL)
    :
      FaceDetectorBase(useOCL),
      m_kw(0.3),
      m_kh(0.3)
{
#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__)
	std::string fileName = "haarcascade_frontalface_alt2.xml";
#else
	std::string fileName = "../HeartRateMeasure/data/haarcascades/haarcascade_frontalface_alt2.xml";
#endif
    if (m_cascade.empty())
    {
		m_cascade.load(fileName);
		if (m_cascade.empty())
		{
			m_cascade.load("../" + fileName);
		}
    }
    assert(!m_cascade.empty());
}

///
/// \brief FaceDetectorHaar::~FaceDetectorHaar
///
FaceDetectorHaar::~FaceDetectorHaar()
{
}

///
/// \brief FaceDetectorHaar::DetectBiggestFace
/// \param image
/// \return
///
cv::Rect FaceDetectorHaar::DetectBiggestFace(cv::UMat image)
{
    cv::Rect res(0, 0, 0, 0);

	if (m_cascade.empty())
	{
		assert(0);
		return res;
	}

    bool findLargestObject = true;
    bool filterRects = true;

    cv::UMat im;
    if (image.channels() == 3)
    {
        cv::cvtColor(image, im, cv::COLOR_BGR2GRAY);
    }
    else
    {
        im = image;
    }

    std::vector<cv::Rect> faceRects;
    m_cascade.detectMultiScale(im,
                               faceRects,
                               1.2,
                               (filterRects || findLargestObject) ? 4 : 0,
                               findLargestObject ? cv::CASCADE_FIND_BIGGEST_OBJECT : 0,
                               cv::Size(image.cols / 4, image.rows / 4));
    if (!faceRects.empty())
    {
        res = faceRects[0];
    }

    return res;
}


///
/// \brief FaceDetectorDNN::FaceDetectorDNN
///
FaceDetectorDNN::FaceDetectorDNN(bool useOCL)
    :
      FaceDetectorBase(useOCL),
      m_confidenceThreshold(0.3)
{
#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__)
    m_modelConfiguration = "deploy.prototxt";
    m_modelBinary = "res10_300x300_ssd_iter_140000.caffemodel";
#else
    m_modelConfiguration = "../HeartRateMeasure/data/face_detector/deploy.prototxt";
    m_modelBinary = "../HeartRateMeasure/data/face_detector/res10_300x300_ssd_iter_140000.caffemodel";
#endif

    m_net = cv::dnn::readNetFromCaffe(m_modelConfiguration, m_modelBinary);

    assert(!m_net.empty());

    if (useOCL)
    {
        m_net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
    }
}

///
/// \brief FaceDetectorDNN::~FaceDetectorDNN
///
FaceDetectorDNN::~FaceDetectorDNN()
{
}

///
/// \brief FaceDetectorDNN::DetectBiggestFace
/// \param image
/// \return
///
cv::Rect FaceDetectorDNN::DetectBiggestFace(cv::UMat image)
{
    cv::Rect res(0, 0, 0, 0);

    const size_t inWidth = 300;
    const size_t inHeight = 300;
    const double inScaleFactor = 1.0;
    const cv::Scalar meanVal(104.0, 177.0, 123.0);

    cv::Mat inputBlob = cv::dnn::blobFromImage(image, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, false, false);

    m_net.setInput(inputBlob, "data");

    cv::Mat detection = m_net.forward("detection_out");

    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    for (int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);

        if (confidence > m_confidenceThreshold)
        {
            int xLeftBottom = cvRound(detectionMat.at<float>(i, 3) * image.cols);
            int yLeftBottom = cvRound(detectionMat.at<float>(i, 4) * image.rows);
            int xRightTop = cvRound(detectionMat.at<float>(i, 5) * image.cols);
            int yRightTop = cvRound(detectionMat.at<float>(i, 6) * image.rows);

            cv::Rect object(xLeftBottom, yLeftBottom, xRightTop - xLeftBottom, yRightTop - yLeftBottom);

            if (object.width > res.width)
            {
                res = object;
            }
        }
    }

    return res;
}
