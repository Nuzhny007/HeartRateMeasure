#include <deque>
#include <iostream>
#include <string>
#include <memory>

#include <opencv2/core/ocl.hpp>

#include "beat_calc/MainProcess.h"
#include "common/common.h"

///
/// \brief main
/// \param argc
/// \param argv
/// \return
///
int main(int argc, char* argv[])
{
	std::cout << "Commmand line arguments (" << argc << "): ";
	for (int i = 0; i < argc; ++i)
	{
		std::cout << argv[i] << " ";
	}
	std::cout << std::endl;

    if (argc < 3)
    {
        std::cerr << "Set video and config files!!!" << std::endl;
        return -1;
    }

	std::string appFullPath(argv[0]);
	std::string appDirPath = appFullPath.substr(0, appFullPath.find_last_of(PathSeparator()));
    
	if (appFullPath == appDirPath)
	{
		appDirPath = "";
	}
	else
	{
		appDirPath += PathSeparator();
	}

	std::cout << appFullPath << ", " << appDirPath << std::endl;

    std::string videoFileName = argv[1];
    std::string confFileName = argv[2];
	std::string confFileNameFull = appDirPath + confFileName;

	MeasureSettings settings;
	if (!settings.ParseOptions(confFileNameFull))
    {
        std::cerr << "Config file \"" << confFileNameFull << "\' is not opened!" << std::endl;
        return -2;
    }

    cv::ocl::setUseOpenCL(settings.m_useOCL);
    std::cout << (cv::ocl::useOpenCL() ? "OpenCL is enabled" : "OpenCL not used") << std::endl;

	// Camera initialization
	settings.m_useFPS = true;
    settings.m_fps = 25;
	settings.m_freq = cv::getTickFrequency();

    cv::VideoCapture capture;
    if (!OpenCapture(videoFileName, capture, settings.m_useFPS, settings.m_freq, settings.m_fps, settings.m_cameraBackend))
    {
        std::cerr << "File or cam not opened!" << std::endl;
        return -1;
    }
    std::cout << "Time frequency = " << settings.m_freq << ", fps = " << settings.m_fps << std::endl;

	int maxFrames = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT));
	std::string outWndName = videoFileName;
	cv::namedWindow(outWndName, cv::WINDOW_NORMAL);

	// Image for plot
    int frameWidth = cvRound(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = cvRound(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    cv::Mat freqPlot(std::min(100, frameHeight), frameWidth / 2, CV_8UC3, cv::Scalar::all(0));
	cv::Mat signalPlot(freqPlot.rows, freqPlot.cols, freqPlot.type());

    cv::VideoWriter videoWiter;
    if (settings.m_saveResults)
    {
        std::string resFileName = videoFileName + "_" + std::to_string(settings.m_sampleSize) + "_" + (settings.m_useMA ? "ma" : "noma") + "_" + settings.m_filterName + "_result.avi";
        videoWiter.open(resFileName, cv::VideoWriter::fourcc('H', 'F', 'Y', 'U'), settings.m_fps, cv::Size(settings.m_useMA ? (2 * frameWidth) : frameWidth, frameHeight), true);
        if (!videoWiter.isOpened())
        {
            std::cerr << "Can't create " << resFileName << std::endl;
        }
        else
        {
            std::cout << resFileName << " was created" << std::endl;
        }
    }
    else
    {
        std::cout << "Without result file" << std::endl;
    }

    MainProcess mainProc(appDirPath);
    mainProc.Init(settings, videoFileName);

    double tick_freq = cv::getTickFrequency();

    bool manual = false;

    int frameInd = 0;
    cv::Mat rgbframe;
    for (capture >> rgbframe; !rgbframe.empty(); capture >> rgbframe)
	{
        int64 t1 = cv::getTickCount();
        int64 captureTime = settings.m_useFPS ? static_cast<int64>((frameInd * 1000.) / settings.m_fps) : t1;

        cv::Mat frame;
        if (mainProc.Process(rgbframe, frame, captureTime, true, true, false, true))
        {
			mainProc.DrawSignal(signalPlot, true, true);
			mainProc.DrawFrequency(freqPlot);
        }
        else
        {
        }

        int64 t2 = cv::getTickCount();

        std::cout << frameInd << ": capture time = " << captureTime << std::endl;

        // Draw color processing result
        frame(cv::Rect(frame.cols - freqPlot.cols, 0, freqPlot.cols, freqPlot.rows)) *= 0.5;
        frame(cv::Rect(frame.cols - freqPlot.cols, 0, freqPlot.cols, freqPlot.rows)) += 0.5 * freqPlot;
		frame(cv::Rect(0, 0, signalPlot.cols, signalPlot.rows)) *= 0.5;
		frame(cv::Rect(0, 0, signalPlot.cols, signalPlot.rows)) += 0.5 * signalPlot;

		int measure = mainProc.RemainingMeasurements();
		if (measure > 0)
		{
			std::string str = "Waiting for " + std::to_string(measure) + " frames";
			cv::putText(frame, str, cv::Point(frame.cols - freqPlot.cols, freqPlot.rows / 3 - 5), cv::FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar::all(255));
		}
		else
		{
			char str[1024];
			FrequencyResults freqResults;
			mainProc.GetFrequency(&freqResults);
			sprintf(str, "[%2.2f, %2.2f] = %2.2f - %2.2f, snr = %2.2f", freqResults.minFreq, freqResults.maxFreq, freqResults.freq, freqResults.smootFreq, freqResults.snr);
			cv::putText(frame, str, cv::Point(frame.cols - freqPlot.cols, freqPlot.rows / 3 - 5), cv::FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar::all(255));
		}

        // Draw detection data
        cv::rectangle(frame, mainProc.GetFaceRect(), cv::Scalar(0, 255, 0), 1);
        const std::vector<cv::Point2f>& currLandmarks = mainProc.GetCurrLandmarks();
        for (auto pt : currLandmarks)
        {
            cv::circle(frame, cv::Point(cvRound(pt.x), cvRound(pt.y)), 2, cv::Scalar(0, 150, 0), 1, cv::LINE_8);
        }

        if (settings.m_useMA)
        {
            cv::Mat outImg(frame.rows, frame.cols * 2, CV_32FC3);
            cv::hconcat(rgbframe, frame, outImg);
            cv::imshow(outWndName, outImg);

            if (videoWiter.isOpened())
            {
                videoWiter << outImg;
            }
        }
        else
        {
            cv::imshow(outWndName, frame);

            if (videoWiter.isOpened())
            {
                videoWiter << frame;
            }
        }

        double t = (t2 - t1) / tick_freq;
        int waitTime = manual ? 0 : (std::max<int>(1, static_cast<int>(1000 / settings.m_fps - t * 1000) - 1));
		std::cout << frameInd << " (" << maxFrames << "): t = " << t << ", waitTime = " << waitTime << std::endl;
        int k = cv::waitKey(waitTime);

        switch (k)
        {
        case 'm':
        case 'M':
            manual = !manual;
            std::cout << "Use manual frame step: " << manual << std::endl;
            break;

        case 27:
            break;
        }

        if (k == 27)
        {
            break;
        }

        ++frameInd;
	}

    cv::waitKey(1000);

    return 0;
}
