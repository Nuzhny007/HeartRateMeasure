#include "FaceDetector.h"

//------------------------------------------------------
//
//------------------------------------------------------
FaceDetector::FaceDetector()
    :
      m_kw(0.3),
      m_kh(0.3)
{
#ifdef USE_GPU
	std::string fileName = "../HeartRateMeasure/data/haarcascades_cuda/haarcascade_frontalface_alt2.xml";
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

//------------------------------------------------------
//
//------------------------------------------------------
FaceDetector::~FaceDetector()
{
#ifdef USE_GPU
    if (!m_cascade.empty())
    {
        m_cascade.release();
    }
#endif
}

//------------------------------------------------------
//
//------------------------------------------------------
cv::Rect FaceDetector::detect_biggest_face(cv::Mat& image, bool originalFace)
{
    cv::Rect res(0, 0, 0, 0);

	if (m_cascade.empty())
	{
		assert(0);
		return res;
	}

    bool findLargestObject = true;
    bool filterRects = true;

    cv::Mat im;
    if (image.channels() == 3)
    {
        cv::cvtColor(image, im, cv::COLOR_BGR2GRAY);
    }
    else
    {
        im = image;
    }

#ifdef USE_GPU
    cv::gpu::GpuMat gray_gpu(im);
    cv::gpu::GpuMat facesBuf_gpu;

    m_cascade.visualizeInPlace = false;
    m_cascade.findLargestObject = findLargestObject;
    int detections_num = m_cascade.detectMultiScale(gray_gpu,
                                                    facesBuf_gpu,
                                                    1.2,
                                                    (filterRects || findLargestObject) ? 4 : 0,
                                                    cv::Size(image.cols / 4, image.rows / 4));

    if (detections_num == 0)
    {
        gray_gpu.release();
        facesBuf_gpu.release();
        return res;
    }

    cv::Mat faces_downloaded;
    facesBuf_gpu.colRange(0, detections_num).download(faces_downloaded);
    cv::Rect *faceRects = faces_downloaded.ptr<cv::Rect>();

    res = faceRects[0];

    gray_gpu.release();
    facesBuf_gpu.release();
#else
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
#endif

    if (!originalFace)
    {
        const double dw = res.width * m_kw;
        const double dh = res.height * m_kh;
        const double dx = dw / 2.0;
        const double dy = dh / 2.0;

        res.x += static_cast<int>(dx);
        res.y += static_cast<int>(dy);
        res.width -= static_cast<int>(dw);
        res.height -= static_cast<int>(dh);
    }

    return res;
}

