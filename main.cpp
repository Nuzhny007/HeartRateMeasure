#include <deque>
#include <iostream>
#include <string>
#include <memory>

#include "SignalProcessorColor.h"
#include "SignalProcessorMoving.h"

#include "detect_track/FaceDetector.h"
#include "detect_track/SkinDetector.h"
#include "detect_track/LKTracker.h"
#include "eulerian_ma/EulerianMA.h"

#include <opencv2/core/ocl.hpp>

double Freq = cv::getTickFrequency();

///
/// \brief The RectSelection enum
///
enum RectSelection
{
    NoneSelection = 0,
    FaceDetection,
    ManualSelection
};

///
/// \brief SkinInit
/// \param skinDetector
/// \return
///
bool SkinInit(SkinDetector& skinDetector)
{
#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__)
	std::string skinPath("");
#else
	std::string skinPath("../HeartRateMeasure/data/");
#endif

	bool res = skinDetector.InitModel(skinPath);

    if (!res)
    {
		res = skinDetector.LearnModel(skinPath);
        if (!res)
        {
            std::cout << "Skin detector wasn't initializad!" << std::endl;
        }
        else
        {
			skinDetector.SaveModel(skinPath);
        }
    }
    return res;
}

///
const char* keys =
{
    "{ @1              |../data/face.mp4    | Video file or web camera index | }"
    "{ s  size         |256                 | Sample size (power of 2) | }"
    "{ ma motion_ampfl |0                   | Use or not motion ampflification | }"
    "{ sd skin         |0                   | Use or not skin detection | }"
    "{ ft filter       |ica                 | Filter type: pca or ica | }"
    "{ g gpu           |0                   | Use OpenCL acceleration | }"
    "{ o out           |0                   | Write result to disk | }"
};

///
/// \brief main
/// \param argc
/// \param argv
/// \return
///
int main(int argc, char* argv[])
{
	// чтобы писать по-русски
	setlocale(LC_ALL, "Russian");

    cv::CommandLineParser parser(argc, argv, keys);

    bool useOCL = parser.get<int>("gpu") != 0;
    cv::ocl::setUseOpenCL(useOCL);
    std::cout << (cv::ocl::useOpenCL() ? "OpenCL is enabled" : "OpenCL not used") << std::endl;

	// Инициализация камеры
    bool useFPS = true;
    double fps = 25;
    std::string fileName = parser.get<std::string>(0);

    // Use motion ampflifacation
    bool useMA = parser.get<int>("motion_ampfl") != 0;

    // Количество измерений в графике
    int sampleSize = parser.get<int>("size");

    bool useSkinDetection = parser.get<int>("skin") != 0;;
    RectSelection selectionType = FaceDetection;

    SignalProcessorColor::RGBFilters filterType = (parser.get<std::string>("filter") == "ica") ? SignalProcessorColor::FilterICA : SignalProcessorColor::FilterPCA;

    cv::VideoCapture capture;
    if (fileName.size() > 1)
    {
        capture.open(fileName);

        if (!capture.isOpened())
        {
            capture.open(0);
            useFPS = false;
            if (!capture.isOpened())
            {
                std::cerr << "File or cam not opened!" << std::endl;
                return -1;
            }
        }
    }
    else
    {
        capture.open(atoi(fileName.c_str()));
        useFPS = false;
        if (!capture.isOpened())
        {
            std::cerr << "File or cam not opened!" << std::endl;
            return -1;
        }
    }

    if (useFPS)
    {
        fps = capture.get(cv::CAP_PROP_FPS);
        Freq = 1000.;
    }
    std::cout << "Time frequency = " << Freq << ", fps = " << fps << std::endl;

	// Создаем окошко
    cv::namedWindow("output", cv::WINDOW_NORMAL);

	// Изображение для вывода графика
    int frameWidth = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    int frameHeight = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    cv::Mat imgColor(std::min(100, frameHeight), std::min(640, frameWidth), CV_8UC3, cv::Scalar::all(0));
    cv::Mat imgMoving(std::min(100, frameHeight), std::min(640, frameWidth), CV_8UC3, cv::Scalar::all(0));

    cv::VideoWriter videoWiter;
    if (parser.get<int>("out") > 0)
    {
        std::string resFileName = fileName + "_" + std::to_string(sampleSize) + "_" + (useMA ? "ma" : "noma") + "_" + parser.get<std::string>("filter") + "_result.avi";
        videoWiter.open(resFileName, CV_FOURCC('H', 'F', 'Y', 'U'), fps, cv::Size(useMA ? (2 * frameWidth) : frameWidth, frameHeight), true);
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

    // Прямоугольник с лицом и точки на нём
    cv::Rect currentRect;
    std::vector<cv::Point2f> currLandmarks;
    std::vector<cv::Point2f> prevLandmarks;

    // Face detector and tracker
    std::unique_ptr<FaceDetectorBase> faceDetector = std::make_unique<FaceDetectorHaar>(useOCL);
    FaceLandmarksDetector landmarksDetector;
    SkinDetector skinDetector;
    if (!SkinInit(skinDetector))
    {
        useSkinDetection = false;
    }

    LKTracker faceTracker;
    EulerianMA eulerianMA;

	// Создаем анализатор
    SignalProcessorColor signalProcessorColor(sampleSize, filterType);
    SignalProcessorMoving signalProcessorMoving(sampleSize);

    double tick_freq = cv::getTickFrequency();

    bool manual = false;

    int frameInd = 0;
    cv::Mat frame;
    cv::Mat rgbframe;
    for (capture >> rgbframe; !rgbframe.empty(); capture >> rgbframe)
	{
		// Запоминаем текущее время
        TimerTimestamp t1 = cv::getTickCount();

        if (useMA)
        {
            if (frameInd == 0)
            {
                eulerianMA.Init(rgbframe, 10, 16, 0.4, 3.0, cvRound(fps), 1.0);
            }
            else
            {
                cv::Mat output = eulerianMA.Process(rgbframe);
                output.convertTo(frame, CV_8UC3, 255);
            }
        }
        else
        {
            frame = rgbframe;
        }
        if (frame.empty())
        {
            ++frameInd;
            continue;
        }

        TimerTimestamp captureTime = useFPS ? ((frameInd * 1000.) / fps) : t1;
        std::cout << frameInd << ": capture time = " << captureTime << std::endl;

        switch (selectionType)
        {
        case NoneSelection:
            currentRect = cv::Rect(0, 0, rgbframe.cols, rgbframe.rows);
            break;

        case FaceDetection:
        {
            cv::UMat uframe = rgbframe.getUMat(cv::ACCESS_READ);

            bool moveSPUpdated = false;

            // Детект лица
            cv::Rect face = faceDetector->DetectBiggestFace(uframe);
            // Tracking
            if (face.area() > 0)
            {
                landmarksDetector.Detect(uframe, face, currLandmarks);
                if (!currLandmarks.empty())
                {
                    faceTracker.ReinitTracker(face, currLandmarks);
                    currentRect = faceTracker.GetTrackedRegion();

                    if (prevLandmarks.size() == currLandmarks.size())
                    {
                        cv::Point2d sum(0, 0);
                        for (size_t i = 0; i < currLandmarks.size(); ++i)
                        {
                            sum.x += currLandmarks[i].x - prevLandmarks[i].x;
                            sum.y += currLandmarks[i].y - prevLandmarks[i].y;
                        }
                        double val = sqrt(sqr(sum.x) + sqr(sum.y));

                        signalProcessorMoving.AddMeasure(captureTime, val);
                        signalProcessorMoving.MeasureFrequency(imgMoving, Freq, frameInd);
                        moveSPUpdated = true;
                        //std::cout << "signalProcessorMoving update by value" << std::endl;
                    }
                }

                prevLandmarks.assign(std::begin(currLandmarks), std::end(currLandmarks));
            }
            if (face.area() == 0)
            {
                faceTracker.Track(rgbframe);
                if (!faceTracker.IsLost())
                {
                    currentRect = faceTracker.GetTrackedRegion();

                    cv::Point2d sum;
                    if (faceTracker.GetMovingSum(sum))
                    {
                        double val = sqrt(sqr(sum.x) + sqr(sum.y));
                        signalProcessorMoving.AddMeasure(captureTime, val);
                        signalProcessorMoving.MeasureFrequency(imgMoving, Freq, frameInd);
                        moveSPUpdated = true;
                        std::cout << "signalProcessorMoving update by tracking" << std::endl;
                    }
                }
                else
                {
                    currentRect = cv::Rect();
                }
            }
            if (currentRect.empty())
            {
                std::cout << "No face!" << std::endl;
            }
            else
            {
                if (currentRect.x < 0)
                {
                    currentRect.x = 0;
                }
                if (currentRect.x + currentRect.width > rgbframe.cols - 1)
                {
                    currentRect.width = rgbframe.cols - 1 - currentRect.x;
                }
                if (currentRect.y < 0)
                {
                    currentRect.y = 0;
                }
                if (currentRect.y + currentRect.height > rgbframe.rows - 1)
                {
                    currentRect.height = rgbframe.rows - 1 - currentRect.y;
                }
                if (!moveSPUpdated)
                {
                    signalProcessorMoving.Reset();
                    std::cout << "signalProcessorMoving.Reset!!!!!!!!!!!" << std::endl;
                }
            }
        }
            break;

        case ManualSelection:
            if (currentRect.empty())
            {
                currentRect = cv::selectROI(rgbframe, true, false);
            }
            break;
        }

		// Если есть объект ненулевой площади вычисляем среднее по цвету
        if (currentRect.area() > 0)
		{
            cv::Mat skinMask;
            if (useSkinDetection)
            {
                skinMask = skinDetector.Detect(rgbframe(currentRect), true, frameInd);
            }
            cv::Scalar meanVal = cv::mean(frame(currentRect), skinMask.empty() ? cv::noArray() : skinMask);

            signalProcessorColor.AddMeasure(captureTime, cv::Vec3d(meanVal.val));
            signalProcessorColor.MeasureFrequency(imgColor, Freq, frameInd);
		}
        else
        {
            signalProcessorColor.Reset();
        }

        // Draw color processing result
        frame(cv::Rect(frame.cols - imgColor.cols, 0, imgColor.cols, imgColor.rows)) *= 0.5;
        frame(cv::Rect(frame.cols - imgColor.cols, 0, imgColor.cols, imgColor.rows)) += 0.5 * imgColor;

		char str[1024];
        double minFreq = 0;
        double maxFreq = 0;
        double currFreq = signalProcessorColor.GetInstantaneousFreq(&minFreq, &maxFreq);
        sprintf(str, "[%2.2f, %2.2f] = %2.2f - %2.2f", minFreq, maxFreq, currFreq, signalProcessorColor.GetFreq());
        cv::putText(frame, str, cv::Point(frame.cols - imgColor.cols, 50), CV_FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar::all(255));

        // Draw moving processing result
        frame(cv::Rect(frame.cols - imgColor.cols - imgMoving.cols, 0, imgMoving.cols, imgMoving.rows)) *= 0.5;
        frame(cv::Rect(frame.cols - imgColor.cols - imgMoving.cols, 0, imgMoving.cols, imgMoving.rows)) += 0.5 * imgMoving;

        minFreq = 0;
        maxFreq = 0;
        currFreq = signalProcessorMoving.GetInstantaneousFreq(&minFreq, &maxFreq);
        sprintf(str, "[%2.2f, %2.2f] = %2.2f - %2.2f", minFreq, maxFreq, currFreq, signalProcessorMoving.GetFreq());
        cv::putText(frame, str, cv::Point(frame.cols - imgColor.cols - imgMoving.cols, 50), CV_FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar::all(255));

        // Draw detection data
        cv::rectangle(frame, currentRect, cv::Scalar(0, 255, 0), 1);
        for (auto pt : currLandmarks)
        {
            cv::circle(frame, cv::Point(cvRound(pt.x), cvRound(pt.y)), 2, cv::Scalar(0, 150, 0), 1, cv::LINE_8);
        }

        if (useMA)
        {
            cv::Mat outImg(frame.rows, frame.cols * 2, CV_32FC3);
            cv::hconcat(rgbframe, frame, outImg);
            cv::imshow("output", outImg);

            if (videoWiter.isOpened())
            {
                videoWiter << outImg;
            }
        }
        else
        {
            cv::imshow("output", frame);

            if (videoWiter.isOpened())
            {
                videoWiter << frame;
            }
        }

        TimerTimestamp t2 = cv::getTickCount();

        double t = (t2 - t1) / tick_freq;
        std::cout << frameInd << ": t = " << t << std::endl;
        int waitTime = manual ? 0 : (std::max<int>(1, 1000 / fps - t / 1000));
        int k = cv::waitKey(waitTime);

        switch (k)
        {
        case 'm':
        case 'M':
            manual = !manual;
            std::cout << "Use manual frame step: " << manual << std::endl;
            break;

        case 'a':
        case 'A':
            useMA = !useMA;
            if (useMA)
            {
                eulerianMA.Init(rgbframe, 10, 16, 0.4, 3.0, 30, 1.0);
            }
            std::cout << "Use motion magnification: " << useMA << std::endl;
            break;

        case 27:
            break;

        case 'r':
        case 'R':
            signalProcessorColor.Reset();
            std::cout << "Reset SignalProcessor" << std::endl;
            break;

        case 's':
        case 'S':
            useSkinDetection = !useSkinDetection;
            std::cout << "Use skin detection: " << useSkinDetection << std::endl;
            break;

        case 'd':
        case 'D':
            switch (selectionType)
            {
            case NoneSelection:
                selectionType = FaceDetection;
                std::cout << "Selection type: without selection" << std::endl;
                break;
            case FaceDetection:
                std::cout << "Selection type: face detection and tracking" << std::endl;
                selectionType = ManualSelection;
                currentRect = cv::Rect(0, 0, 0, 0);
                break;
            case ManualSelection:
                std::cout << "Selection type: manual rectangle selection" << std::endl;
                selectionType = NoneSelection;
                break;
            }

            break;
        }
        if (k == 27)
        {
            break;
        }

        ++frameInd;
	}

    cv::waitKey(0);

    return 0;
}
