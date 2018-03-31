#include <deque>
#include <iostream>
#include <string>
#include <memory>

#include "SignalProcessor.h"

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

    SignalProcessor::RGBFilters filterType = (parser.get<std::string>("filter") == "ica") ? SignalProcessor::FilterICA : SignalProcessor::FilterPCA;

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
    cv::Mat I(std::min(100, frameHeight), std::min(640, frameWidth), CV_8UC3, cv::Scalar::all(0));

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
    std::vector<cv::Point2f> landmarks;

    // Face detector and tracker
    std::unique_ptr<FaceDetectorBase> faceDetector = std::make_unique<FaceDetectorDNN>(useOCL);
    FaceLandmarksDetector landmarksDetector;
    SkinDetector skinDetector;
    if (!SkinInit(skinDetector))
    {
        useSkinDetection = false;
    }

    LKTracker faceTracker;
    EulerianMA eulerianMA;

	// Создаем анализатор
    SignalProcessor sp(sampleSize, filterType);

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
                eulerianMA.Init(rgbframe, 10, 16, 0.4, 3.0, 30, 1.0);
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

        switch (selectionType)
        {
        case NoneSelection:
            currentRect = cv::Rect(0, 0, rgbframe.cols, rgbframe.rows);
            break;

        case FaceDetection:
        {
            cv::UMat uframe = rgbframe.getUMat(cv::ACCESS_READ);

            // Детект лица
            cv::Rect face = faceDetector->DetectBiggestFace(uframe);
            // Tracking
            if (face.area() > 0)
            {
                landmarksDetector.Detect(uframe, face, landmarks);
                if (!landmarks.empty())
                {
                    faceTracker.ReinitTracker(face, landmarks);
                    currentRect = faceTracker.GetTrackedRegion();
                }
            }
            if (face.area() == 0)
            {
                faceTracker.Track(rgbframe);
                if (!faceTracker.IsLost())
                {
                    currentRect = faceTracker.GetTrackedRegion();
                }
                else
                {
                    currentRect = cv::Rect();
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

            TimerTimestamp captureTime = useFPS ? ((frameInd * 1000.) / fps) : t1;
            std::cout << frameInd << ": capture time = " << captureTime << std::endl;
            sp.AddMeasure(captureTime, cv::Vec3d(meanVal.val));
            sp.MeasureFrequency(I, Freq, frameInd);
		}
        else
        {
            sp.Reset();
        }

        frame(cv::Rect(frame.cols - I.cols, 0, I.cols, I.rows)) *= 0.5;
        frame(cv::Rect(frame.cols - I.cols, 0, I.cols, I.rows)) += 0.5 * I;

        cv::rectangle(frame, currentRect, cv::Scalar(0, 255, 0), 1);

		char str[1024];
        double minFreq = 0;
        double maxFreq = 0;
        double currFreq = sp.GetInstantaneousFreq(&minFreq, &maxFreq);
        sprintf(str, "[%2.2f, %2.2f] = %2.2f - %2.2f", minFreq, maxFreq, currFreq, sp.GetFreq());
        cv::putText(frame, str, cv::Point(frame.cols - I.cols, 50), CV_FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar::all(255));

        for (auto pt : landmarks)
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
        std::cout << "t = " << t << std::endl;
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
            sp.Reset();
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
