#pragma once

#include <deque>
#include <iostream>
#include <string>
#include <memory>

#include "SignalPlugin.h"

#include "../detect_track/FaceDetector.h"
#include "../detect_track/SkinDetector.h"
#include "../detect_track/LKTracker.h"
#include "../eulerian_ma/MotionAmp.h"
#include "../common/common.h"

#include <opencv2/core/ocl.hpp>
#include <opencv2/tracking.hpp>

#define USE_LK_TRACKER 0

///
/// \brief The MainProcess class
///
class MainProcess
{
public:
    MainProcess(const std::string& appDirPath);
    ~MainProcess();

    bool Init(const MeasureSettings& settings, const std::string& videoName);
    bool Process(cv::Mat rgbFrame, cv::Mat& imgProc, int64 captureTime, cv::Scalar& colorVal, bool drawResults, bool saveResults, bool createResultsPanno, bool showMixture);

    cv::Rect GetFaceRect() const;
    void GetFrequency(FrequencyResults* freqResults, double* meanFreq = nullptr, double* devFreq = nullptr);
    const std::vector<cv::Point2f>& GetCurrLandmarks() const;
	int RemainingMeasurements();

	bool DrawSignal(cv::Mat& signalPlot, bool drawSignal, bool saveSignal);
	bool DrawFrequency(cv::Mat& freqPlot);

private:
	std::string m_appDirPath;
    cv::Rect m_currFaceRect;
    std::vector<cv::Point2f> m_prevLandmarks;
	cv::Mat m_prevFrame;

	StatisticLogger<double> m_measureLogger;

    std::unique_ptr<FaceDetectorBase> m_faceDetector;
    SkinDetector m_skinDetector;

#if USE_LK_TRACKER
	LKTracker m_faceTracker;
	FaceLandmarksDetector m_landmarksDetector;
#else
	cv::Ptr<cv::Tracker> m_faceTracker;
#endif
	FaceCrop m_faceCrop;

	SignalPlugin m_signalProcessorColor;

    std::unique_ptr<MotionAmp> m_eulerianMA;

    int m_frameInd = 0;

    MeasureSettings m_settings;

	cv::Mat m_motionMap;
	cv::Mat m_faceMask;
	void CalcMotionMap(cv::Mat frame, cv::Mat skinMask, const cv::Rect& faceRect);
	void DrawResult(cv::Mat frame, const cv::Rect& faceRect, const cv::Rect& resultFaceRect, const std::vector<cv::Point2f>& landmarks);

	bool TrackFace(cv::Mat rgbFrame);
};
