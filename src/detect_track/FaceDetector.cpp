#include "FaceDetector.h"
#include "../common/common.h"

///
/// \brief FaceDetectorHaar::FaceDetectorHaar
///
FaceDetectorHaar::FaceDetectorHaar(const std::string& appDirPath, bool useOCL)
    :
      FaceDetectorBase(appDirPath, useOCL),
      m_kw(0.3),
      m_kh(0.3)
{
#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__)
	std::string fileName = appDirPath + "data" + PathSeparator() + "haarcascade_frontalface_alt2.xml";
#else
    std::string fileName = "../beatmagnifier/data/haarcascades/haarcascade_frontalface_alt2.xml";
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
                               1.1,
                               (filterRects || findLargestObject) ? 3 : 0,
                               findLargestObject ? cv::CASCADE_FIND_BIGGEST_OBJECT : 0,
                               cv::Size(image.cols / 8, image.rows / 8));
    if (!faceRects.empty())
    {
        res = faceRects[0];
    }

    return res;
}


///
/// \brief FaceDetectorDNN::FaceDetectorDNN
///
FaceDetectorDNN::FaceDetectorDNN(const std::string& appDirPath, bool useOCL, bool useOpenVINO)
    :
      FaceDetectorBase(appDirPath, useOCL),
      m_confidenceThreshold(0.3f)
{
	std::string weightFile;
	std::string confFile;
	if (useOpenVINO)
	{
#if 0
		weightFile = "face-detection-adas-0001.bin";
		confFile = "face-detection-adas-0001.xml";
#else
		weightFile = "face-detection-retail-0004.bin";
		confFile = "face-detection-retail-0004.xml";
#endif
	}
	else
	{
		weightFile = "deploy.prototxt";
		confFile = "res10_300x300_ssd_iter_140000.caffemodel";
	}

#if 0
#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__)
	m_modelConfiguration = appDirPath + "data" + PathSeparator() + confFile;
	m_modelBinary = appDirPath + "data" + PathSeparator() + weightFile;
#else
    m_modelConfiguration = "../beatmagnifier/data/face_detector/" + confFile;
    m_modelBinary = "../beatmagnifier/data/face_detector/" + weightFile;
#endif
#else
    m_modelConfiguration = appDirPath + "data" + PathSeparator() + confFile;
    m_modelBinary = appDirPath + "data" + PathSeparator() + weightFile;
#endif
    m_net = cv::dnn::readNet(m_modelBinary, m_modelConfiguration);

    assert(!m_net.empty());
	
	m_net.setPreferableBackend(useOpenVINO ? cv::dnn::DNN_BACKEND_INFERENCE_ENGINE : cv::dnn::DNN_BACKEND_OPENCV);
	
	m_net.setPreferableTarget(useOCL ? cv::dnn::DNN_TARGET_OPENCL : cv::dnn::DNN_TARGET_CPU);
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

#if 0
    for (int i = 0; i < 4; ++i)
    {
        std::cout << "detection.size[" << i << "] = " << detection.size[i] << "; ";
    }
    std::cout << std::endl;
#endif

    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    for (int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);
        int objectClass = cvRound(detectionMat.at<float>(i, 2));

        if (confidence > m_confidenceThreshold && objectClass == 1)
        {
            int xLeftBottom = cvRound(detectionMat.at<float>(i, 3) * image.cols);
            int yLeftBottom = cvRound(detectionMat.at<float>(i, 4) * image.rows);
            int xRightTop = cvRound(detectionMat.at<float>(i, 5) * image.cols);
            int yRightTop = cvRound(detectionMat.at<float>(i, 6) * image.rows);

            cv::Rect object(xLeftBottom, yLeftBottom, xRightTop - xLeftBottom, yRightTop - yLeftBottom);

            if (object.x >=0 && object.y >= 0 &&
                    object.x + object.width < image.rows &&
                    object.y + object.height < image.cols)
            {
                if (object.width > res.width)
                {
                    res = object;
                }
            }
        }

        //std::cout << "Face " << i << ": confidence = " << confidence << ", class = " << cvRound(detectionMat.at<float>(i, 2));
        //std::cout << ", rect(" << detectionMat.at<float>(i, 3) << ", " << detectionMat.at<float>(i, 4) << ", ";
        //std::cout << detectionMat.at<float>(i, 5) << ", " << detectionMat.at<float>(i, 6) << ")" << std::endl;
    }

    return res;
}

///
/// \brief FaceLandmarksDetector::FaceLandmarksDetector
///
FaceLandmarksDetector::FaceLandmarksDetector(const std::string& appDirPath)
{
#if 0
#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__)
	m_modelFilename = appDirPath + "data" + PathSeparator() + "face_landmark_model.dat";
#else
    m_modelFilename = "../beatmagnifier/data/face_detector/face_landmark_model.dat";
#endif
#else
    m_modelFilename = appDirPath + "data" + PathSeparator() + "face_landmark_model.dat";
#endif
    cv::face::FacemarkKazemi::Params params;
    m_facemark = cv::face::FacemarkKazemi::create(params);
    m_facemark->loadModel(m_modelFilename);
}

///
/// \brief FaceLandmarksDetector::~FaceLandmarksDetector
///
FaceLandmarksDetector::~FaceLandmarksDetector()
{

}

///
/// \brief FaceLandmarksDetector::Detect
/// \param image
/// \param faceRect
/// \param landmarks
///
void FaceLandmarksDetector::Detect(cv::UMat image,
                                   const cv::Rect& faceRect,
                                   std::vector<cv::Point2f>& landmarks)
{
    std::vector<cv::Rect> faces = { faceRect };
    std::vector<std::vector<cv::Point2f>> shapes;

    landmarks.clear();

    if (m_facemark->fit(image, faces, shapes))
    {
        landmarks.assign(std::begin(shapes[0]), std::end(shapes[0]));
    }
}

///
FaceDetectorBase* CreateFaceDetector(
	MeasureSettings::FaceDetectors detectorType,
	const std::string& appDirPath,
	bool useOCL)
{
	FaceDetectorBase* faceDetector = nullptr;

	std::cout << "Create face detector: " << detectorType << ", appDirPath = " << appDirPath << ", useOpenCL = " << useOCL << std::endl;

	switch (detectorType)
	{
	case MeasureSettings::Haar:
	{
		FaceDetectorHaar* haarDetector = new FaceDetectorHaar(appDirPath, useOCL);
		faceDetector = haarDetector;
		break;
	}
	case MeasureSettings::Resnet:
	{
		FaceDetectorDNN* dnnDetector = new FaceDetectorDNN(appDirPath, useOCL, false);
		faceDetector = dnnDetector;
		break;
	}

	case MeasureSettings::VINO:
	{
		FaceDetectorDNN* dnnDetector = new FaceDetectorDNN(appDirPath, useOCL, true);
		faceDetector = dnnDetector;
		break;
	}
	}

	return faceDetector;
}
