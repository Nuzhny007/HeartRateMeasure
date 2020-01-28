#include "MainProcess.h"
#include "../eulerian_ma/EulerianMA.h"
#include "../eulerian_ma/SimpleMA.h"

///
/// \brief MedianMat
///
double MedianMat(cv::Mat Input, int nVals)
{
	// COMPUTE HISTOGRAM OF SINGLE CHANNEL MATRIX
	float range[] = { 0, static_cast<float>(nVals) };
	const float* histRange = { range };
	bool uniform = true;
	bool accumulate = false;
	cv::Mat hist;
	calcHist(&Input, 1, 0, cv::Mat(), hist, 1, &nVals, &histRange, uniform, accumulate);

	// COMPUTE CUMULATIVE DISTRIBUTION FUNCTION (CDF)
	cv::Mat cdf;
	hist.copyTo(cdf);
	for (int i = 1; i <= nVals - 1; i++)
	{
		cdf.at<float>(i) += cdf.at<float>(i - 1);
	}
	cdf /= static_cast<float>(Input.total());

	// COMPUTE MEDIAN
	double medianVal = 0;
	for (int i = 0; i <= nVals - 1; i++)
	{
		if (cdf.at<float>(i) >= 0.5)
		{
			medianVal = i;
			break;
		}
	}
	return medianVal / nVals;
}

///
/// \brief MainProcess::MainProcess
///
MainProcess::MainProcess(const std::string& appDirPath)
	:
	m_appDirPath(appDirPath + PathSeparator())
#if USE_LK_TRACKER
	, m_landmarksDetector(appDirPath + PathSeparator())
#endif
{

}

///
/// \brief MainProcess::~MainProcess
///
MainProcess::~MainProcess()
{
#if !USE_LK_TRACKER
	if (m_faceTracker && !m_faceTracker.empty())
	{
		m_faceTracker.release();
	}
#endif

	if (m_signalProcessorColor.IsLoaded())
	{
		m_signalProcessorColor.UnloadPlugin();
	}
}

///
/// \brief MainProcess::Init
/// \param settings
/// \return
///
bool MainProcess::Init(const MeasureSettings& settings, const std::string& videoName)
{
    m_settings = settings;

    cv::ocl::setUseOpenCL(m_settings.m_useOCL);
    std::cout << (cv::ocl::useOpenCL() ? "OpenCL is enabled" : "OpenCL not used") << std::endl;

	m_faceDetector = std::unique_ptr<FaceDetectorBase>(CreateFaceDetector(m_settings.m_faceDetectorType, m_appDirPath, m_settings.m_useOCL));

#if 0
#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__)
    if (!SkinInit(m_skinDetector, m_appDirPath))
#else
    if (!SkinInit(m_skinDetector, "../beatmagnifier/data/"))
#endif
#else
    if (!SkinInit(m_skinDetector, m_appDirPath + "data" + PathSeparator()))
#endif
    {
        m_settings.m_useSkinDetection = false;
    }

	if (m_signalProcessorColor.LoadPlugin(m_settings.m_signalLib))
	{
		InputParams inputParams;
		inputParams.framesCount = m_settings.m_sampleSize;
		inputParams.filterType = m_settings.m_filterType;
		inputParams.signalNormalization = m_settings.m_signalNormalization;
		inputParams.gauss_def_var = m_settings.m_gauss_def_var;
		inputParams.gauss_min_var = m_settings.m_gauss_min_var;
		inputParams.gauss_max_var = m_settings.m_gauss_max_var;
		inputParams.gauss_eps = m_settings.m_gauss_eps;
		inputParams.gauss_update_alpha = m_settings.m_gauss_update_alpha;
		inputParams.gauss_proc_alpha = m_settings.m_gauss_proc_alpha;
		inputParams.gauss_proc_weight_thresh = m_settings.m_gauss_proc_weight_thresh;
		inputParams.retExpFreq = m_settings.m_return_exp_frequency;
		inputParams.fps = static_cast<float>(m_settings.m_fps);

		m_signalProcessorColor.Init(&inputParams);
	}
    
	switch (settings.m_maAlgorithm)
	{
	case MeasureSettings::Unknown:
	case MeasureSettings::Eulerian:
		m_eulerianMA = std::make_unique<EulerianMA>();
		break;

	case MeasureSettings::Simple:
		m_eulerianMA = std::make_unique<SimpleMA>();
		break;
	}

	if (!videoName.empty())
	{
		std::string maName = settings.m_useMA ? ((settings.m_maAlgorithm == MeasureSettings::Simple) ? "sma" : "ma") : "raw";
		std::string fileName = videoName + "_" + std::to_string(settings.m_fps) + "_" + maName + ".csv";
		//m_signalProcessorColor->SaveColorsToFile(fileName);
		m_measureLogger.Init(videoName + "_measurements.csv");
	}

	m_frameInd = 0;
    return true;
}

///
/// \brief MainProcess::Process
/// \param rgbframe
/// \param freqPlot
/// \param captureTime
/// \return
///
bool MainProcess::Process(
        cv::Mat rgbFrame,
        cv::Mat& imgProc,
        int64 captureTime,
	    bool drawResults,
	    bool saveResults,
	    bool createResultsPanno,
	    bool showMixture)
{
    cv::UMat uframe = rgbFrame.getUMat(cv::ACCESS_READ);

    // Детект лица
    cv::Rect face = m_faceDetector->DetectBiggestFace(uframe);
    // Tracking
    if (m_currFaceRect.area() > 0)
    {
		if (face.area() == 0)
		{
			TrackFace(rgbFrame);
		}
		else
		{
			float iou = (face & m_currFaceRect).area() / static_cast<float>((face | m_currFaceRect).area());
			if (iou < 0.4f)
			{
				TrackFace(rgbFrame);
				face = cv::Rect();
			}
		}
    }
	if (face.area() > 16)
	{
		m_currFaceRect = face;
		m_prevLandmarks.clear();
	}
    if (m_currFaceRect.empty())
    {
        std::cout << "No face!" << std::endl;

#if !USE_LK_TRACKER
		if (m_faceTracker && !m_faceTracker.empty())
		{
			m_faceTracker.release();
		}
#endif
    }
    else
    {
        if (m_currFaceRect.x < 0)
        {
            m_currFaceRect.x = 0;
        }
        if (m_currFaceRect.x + m_currFaceRect.width > rgbFrame.cols - 1)
        {
            m_currFaceRect.width = rgbFrame.cols - 1 - m_currFaceRect.x;
        }
        if (m_currFaceRect.y < 0)
        {
            m_currFaceRect.y = 0;
        }
        if (m_currFaceRect.y + m_currFaceRect.height > rgbFrame.rows - 1)
        {
            m_currFaceRect.height = rgbFrame.rows - 1 - m_currFaceRect.y;
        }
    }

	if (m_settings.m_useMA)
	{
		//std::cout << "Start MA" << std::endl;
		if (m_settings.m_maUseCrop && m_currFaceRect.area() > 0)
		{
			//std::cout << "New face" << std::endl;
			cv::Rect crop = m_faceCrop.NewFace(m_currFaceRect, rgbFrame.size());

			//std::cout << "Face rect = " << m_currFaceRect << ", MA crop = " << crop << ", frame size = " << rgbFrame.size() << std::endl;

			if (!m_eulerianMA->IsInitialized() || m_eulerianMA->GetSize() != crop.size())
			{
				//std::cout << "MA init" << std::endl;

				m_eulerianMA->Init(uframe(crop),
					m_settings.m_maAlpha, m_settings.m_maLambdaC,
					m_settings.m_maFlow, m_settings.m_maFhight,
					cvRound(m_settings.m_fps), m_settings.m_maChromAttenuation);
				imgProc = rgbFrame;
			}
			else
			{
				//std::cout << "MA Process" << std::endl;

				cv::UMat output = m_eulerianMA->Process(uframe(crop));
				rgbFrame.copyTo(imgProc);
				output.convertTo(imgProc(crop), CV_8UC3);
			}
		}
		else
		{
			if (!m_eulerianMA->IsInitialized() || m_eulerianMA->GetSize() != uframe.size())
			{
				//std::cout << "MA init" << std::endl;

				m_eulerianMA->Init(uframe,
					m_settings.m_maAlpha, m_settings.m_maLambdaC,
					m_settings.m_maFlow, m_settings.m_maFhight,
					cvRound(m_settings.m_fps), m_settings.m_maChromAttenuation);
				imgProc = rgbFrame;
			}
			else
			{
				//std::cout << "MA Process" << std::endl;

				cv::UMat output = m_eulerianMA->Process(uframe);
				output.convertTo(imgProc, CV_8UC3);
			}
		}
	}
	else
	{
		imgProc = rgbFrame;
	}

    // Если есть объект ненулевой площади вычисляем среднее по цвету
    if (m_currFaceRect.area() > 0)
    {
		//std::cout << "Skin detection" << std::endl;
        cv::Mat skinMask;
        if (m_settings.m_useSkinDetection)
        {
            skinMask = m_skinDetector.Detect(rgbFrame(m_currFaceRect), drawResults, saveResults, m_frameInd);
        }
		//std::cout << "Skin mean" << std::endl;

		cv::Scalar colorVal;
		if (m_settings.m_calcMean)
		{
			colorVal = cv::mean(imgProc(m_currFaceRect), skinMask.empty() ? cv::noArray() : skinMask);
		}
		else
		{
			std::vector<cv::Mat> chans;
			cv::split(imgProc(m_currFaceRect), chans);
			colorVal[0] = MedianMat(chans[0], 256);
			colorVal[1] = MedianMat(chans[1], 256);
			colorVal[2] = MedianMat(chans[2], 256);
		}
		//std::cout << "SP add measure" << std::endl;
        m_signalProcessorColor.AddMeasure(captureTime, colorVal.val);
		//std::cout << "SP measure" << std::endl;
		m_signalProcessorColor.MeasureFrequency(m_settings.m_freq, m_frameInd, showMixture);
		if (createResultsPanno)
		{
			//std::cout << "Calc mm" << std::endl;
			if (!skinMask.empty())
			{
				CalcMotionMap(rgbFrame, skinMask, m_currFaceRect);
			}
			//std::cout << "Draw result" << std::endl;
			DrawResult(rgbFrame, face, m_currFaceRect, m_prevLandmarks);
		}
	}
    else
    {
        m_signalProcessorColor.Reset();
    }

    ++m_frameInd;

	m_prevFrame = rgbFrame;

    return m_currFaceRect.area() > 0;
}

///
/// \brief MainProcess::DrawSignal
/// \param signalPlot
///
bool MainProcess::DrawSignal(cv::Mat& signalPlot, bool drawSignal, bool saveSignal)
{
	SignalInfo signalInfo;
	bool res = m_signalProcessorColor.GetSignal(&signalInfo);
	if (res && signalInfo.m_signal[0])
	{
		std::vector<cv::Mat> signal;
		for (size_t i = 0; i < 3; ++i)
		{
			if (signalInfo.m_signal[i] && signalInfo.m_signalSize[i])
			{
				signal.push_back(cv::Mat(signalInfo.m_signalSize[i], 1, CV_64FC1, signalInfo.m_signal[i]));
			}
		}

		int plotHeight = signalPlot.rows / static_cast<int>(signal.size());

		signalPlot.setTo(0);
		int thikness = 2;
		for (size_t si = 0; si < signal.size(); ++si)
		{
			cv::Mat snorm;
			cv::normalize(signal[si], snorm, plotHeight, 0, cv::NORM_MINMAX);

			double timeSum = 0;
			double v0 = snorm.at<double>(0);
			for (int i = 1; i < snorm.rows; ++i)
			{
				double v1 = snorm.at<double>(i);

				cv::Point pt0(((i - 1) * signalPlot.cols) / snorm.rows, cvRound((si + 1) * plotHeight - v0));
				cv::Point pt1((i * signalPlot.cols) / snorm.rows, cvRound((si + 1) * plotHeight - v1));

				cv::line(signalPlot, pt0, pt1, cv::Scalar(0, 200, 0), thikness);

				int dtPrev = static_cast<int>(1000. * timeSum) / 1000;
				timeSum += signalInfo.m_deltaTime;
				int dtCurr = static_cast<int>(1000. * timeSum) / 1000;
				if (dtCurr > dtPrev)
				{
					cv::line(signalPlot,
						cv::Point(pt1.x, static_cast<int>(si * plotHeight)),
						cv::Point(pt1.x, static_cast<int>((si + 1) * plotHeight - 1)),
						cv::Scalar(0, 150, 0), 1);
				}

				v0 = v1;
			}
			cv::line(signalPlot,
				cv::Point(0, static_cast<int>((si + 1) * plotHeight)),
				cv::Point(signalPlot.cols - 1, static_cast<int>((si + 1) * plotHeight)), cv::Scalar(0, 0, 0));
		}

		if (!signalPlot.empty())
		{
			if (drawSignal)
			{
				cv::namedWindow("signal color", cv::WINDOW_AUTOSIZE);
				cv::imshow("signal color", signalPlot);
			}
			if (saveSignal)
			{
				std::string fileName = "signal_color/" + std::to_string(m_frameInd) + ".png";
				cv::imwrite(fileName, signalPlot);
			}
		}
	}
	else
	{
		res = false;
	}
	return res;
}

///
/// \brief MainProcess::DrawFrequency
/// \param freqPlot
///
bool MainProcess::DrawFrequency(cv::Mat& freqPlot)
{
	bool res = false;

	if (!freqPlot.empty())
	{
		freqPlot.setTo(0);

		SignalInfo signalInfo;
		res = m_signalProcessorColor.GetSignal(&signalInfo);
		if (res && signalInfo.m_spectrum[0])
		{
			int freqD = signalInfo.m_toInd[0] - signalInfo.m_fromInd[0];
			double scale_x = (double)freqPlot.cols / (double)freqD;
			double scale_y = freqPlot.rows / 2.0;

			// Изобразим спектр Фурье
			for (int x = signalInfo.m_fromInd[0]; x < signalInfo.m_toInd[0]; ++x)
			{
				bool findInd = false;

				/*for (auto i : inds)
				{
					if (i == x)
					{
						findInd = true;
						break;
					}
				}*/

				cv::Rect drawRect(cvRound(scale_x * (x - signalInfo.m_fromInd[0])),
					cvRound(freqPlot.rows - scale_y - scale_y * signalInfo.m_spectrum[0][x]),
					cvRound(scale_x),
					std::max(1, cvRound(scale_y * signalInfo.m_spectrum[0][x])));
				cv::rectangle(freqPlot, drawRect, findInd ? cv::Scalar(200, 0, 200) : cv::Scalar(200, 200, 200), cv::FILLED);
			}

			int lastVal = 0;
			int lastTextX = 0;
			for (int x = signalInfo.m_fromInd[0]; x < signalInfo.m_toInd[0]; ++x)
			{
				int currVal = signalInfo.m_freqValues[0][x - signalInfo.m_fromInd[0]];
				if (currVal != lastVal)
				{
					cv::String text = std::to_string(currVal);
					int fontFace = cv::FONT_HERSHEY_TRIPLEX;
					double fontScale = 0.6;
					int thickness = 1;
					int baseLine = 0;
					cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseLine);
					if (textSize.height > freqPlot.rows / 5)
					{
						fontScale = 0.4;
						textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseLine);
					}
					cv::Point textPt(cvRound(scale_x * (x - signalInfo.m_fromInd[0]) + 1), freqPlot.rows - 5);

					if (textPt.x > lastTextX)
					{
						cv::putText(freqPlot, text, textPt, fontFace, fontScale, cv::Scalar::all(255));
						lastTextX = textPt.x + textSize.width + 2;
						lastVal = currVal;
					}
				}
			}
		}
	}
	return res;
}

///
/// \brief MainProcess::DrawResult
/// \param frame
///
void MainProcess::DrawResult(cv::Mat frame,
	const cv::Rect& faceRect,
	const cv::Rect& resultFaceRect,
	const std::vector<cv::Point2f>& landmarks)
{
	cv::rectangle(frame, faceRect, cv::Scalar(0, 0, 200), 1, cv::LINE_AA, 0);
	cv::rectangle(frame, resultFaceRect, cv::Scalar(0, 200, 0), 1, cv::LINE_AA, 0);

	for (auto pt : landmarks)
	{
		cv::circle(frame, cv::Point(cvRound(pt.x), cvRound(pt.y)), 2, cv::Scalar(0, 150, 0), 1, cv::LINE_8);
	}
}

///
/// \brief MainProcess::CalcMotionMap
/// \param frame
///
void MainProcess::CalcMotionMap(cv::Mat frame,
	cv::Mat skinMask,
	const cv::Rect& faceRect)
{
#if 1
	const int chans = frame.channels();

	for (int y = 0; y < faceRect.height; ++y)
	{
		uchar* imgPtr = frame.ptr(faceRect.y + y) + chans * faceRect.x;
		const uchar* maskPtr = skinMask.ptr(y);
		for (int x = 0; x < faceRect.width; ++x)
		{
			if (*maskPtr)
			{
				imgPtr[chans - 1] = (imgPtr[chans - 1] + 255) / 2;
			}
			imgPtr += chans;
			++maskPtr;
		}
	}
#else
	if (m_motionMap.size() != frame.size())
	{
		m_motionMap = cv::Mat(frame.size(), CV_32FC1, cv::Scalar(0, 0, 0));
		m_faceMask = cv::Mat(frame.size(), CV_32FC1, cv::Scalar(0, 0, 0));
	}

	cv::Mat normFor;
	cv::normalize(skinMask, normFor, 255, 0, cv::NORM_MINMAX, m_motionMap.type());
	m_faceMask = cv::Scalar(0);
	normFor.copyTo(m_faceMask(faceRect));

	double alpha = 0.95;
	cv::addWeighted(m_motionMap, alpha, m_faceMask, 1 - alpha, 0, m_motionMap);

	const int chans = frame.channels();

	for (int y = 0; y < frame.rows; ++y)
	{
		uchar* imgPtr = frame.ptr(y);
		float* moPtr = reinterpret_cast<float*>(m_motionMap.ptr(y));
		for (int x = 0; x < frame.cols; ++x)
		{
			for (int ci = chans - 1; ci < chans; ++ci)
			{
				imgPtr[ci] = cv::saturate_cast<uchar>(imgPtr[ci] + moPtr[0]);
			}
			imgPtr += chans;
			++moPtr;
		}
	}
#endif
}

///
/// \brief MainProcess::GetFaceRect
/// \return
///
cv::Rect MainProcess::GetFaceRect() const
{
    return m_currFaceRect;
}

///
/// \brief MainProcess::GetFrequency
///
void MainProcess::GetFrequency(FrequencyResults* freqResults, double* meanFreq, double* devFreq)
{
	m_signalProcessorColor.GetFrequency(freqResults);
	m_measureLogger.NewMeasure(m_frameInd, freqResults->smootFreq);
	if (meanFreq && devFreq)
	{
		m_measureLogger.GetMeanStdDev(*meanFreq, *devFreq);
	}
}

///
/// \brief MainProcess::GetCurrLandmarks
/// \return
///
const std::vector<cv::Point2f>& MainProcess::GetCurrLandmarks() const
{
    return m_prevLandmarks;
}

///
/// \brief MainProcess::RemainingMeasurements
/// \return
///
int MainProcess::RemainingMeasurements()
{
	return m_signalProcessorColor.RemainingMeasurements();
}

///
/// \brief MainProcess::TrackFace
/// \return
///
bool MainProcess::TrackFace(cv::Mat rgbFrame)
{
#if USE_LK_TRACKER
#if 1
	if (m_prevLandmarks.empty())
	{
		std::vector<cv::Point2f> landmarks;
		m_faceTracker.ReinitTracker(m_currFaceRect, landmarks);
		m_faceTracker.Track(m_prevFrame);
	}

	m_currFaceRect = cv::Rect();
	m_faceTracker.Track(rgbFrame);
	if (!m_faceTracker.IsLost())
	{
		m_currFaceRect = m_faceTracker.GetTrackedRegion();
		m_faceTracker.GetPoints(m_prevLandmarks);
	}
#else
	if (m_prevLandmarks.empty())
	{
		cv::UMat uPrevFrame = m_prevFrame.getUMat(cv::ACCESS_READ);
		m_landmarksDetector.Detect(uPrevFrame, m_currFaceRect, m_prevLandmarks);
		if (!m_prevLandmarks.empty())
		{
			m_faceTracker.ReinitTracker(m_currFaceRect, m_prevLandmarks);
			m_faceTracker.Track(m_prevFrame);
		}
	}
	m_currFaceRect = cv::Rect();
	if (!m_prevLandmarks.empty())
	{
		m_faceTracker.Track(rgbFrame);
		if (!m_faceTracker.IsLost())
		{
			m_currFaceRect = m_faceTracker.GetTrackedRegion();
		}
		else
		{
			m_prevLandmarks.clear();
		}
	}
#endif
#else
	if (!m_faceTracker || m_faceTracker.empty())
	{
		cv::TrackerKCF::Params params;
		params.compressed_size = 1;
		params.desc_pca = cv::TrackerKCF::CN;
		params.desc_npca = cv::TrackerKCF::CN;
		params.resize = true;
		params.detect_thresh = 0.5f;
		m_faceTracker = cv::TrackerKCF::create(params);

		m_faceTracker->init(m_prevFrame, cv::Rect2d(m_currFaceRect.x, m_currFaceRect.y, m_currFaceRect.width, m_currFaceRect.height));
	}
	else
	{
		cv::Rect2d newRect;
		if (m_faceTracker->update(rgbFrame, newRect))
		{
			m_currFaceRect.x = static_cast<int>(newRect.x);
			m_currFaceRect.y = static_cast<int>(newRect.y);
			m_currFaceRect.width = static_cast<int>(newRect.width);
			m_currFaceRect.height = static_cast<int>(newRect.height);
		}
	}
#endif
	return !m_currFaceRect.empty();
}
