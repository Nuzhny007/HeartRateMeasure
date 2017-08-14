#include <deque>
#include <iostream>
#include <string>

#include "SignalProcessor.h"

#include "detect_track/FaceDetector.h"
#include "detect_track/SkinDetector.h"
#include "detect_track/LKTracker.h"
#include "eulerian_ma/EulerianMA.h"


double Freq = cv::getTickFrequency();

// --------------------------------------------------------
// 
// --------------------------------------------------------
int main(int argc, char* argv[])
{
#ifdef USE_GPU
    // инициализация GPU
    cv::cuda::setDevice(0);
    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
#endif

	// чтобы писать по-русски
	setlocale(LC_ALL, "Russian");

	// Инициализация камеры
    bool useFPS = true;
    double fps = 25;
    std::string fileName = "../data/face.mp4";

    // Use motion ampflifacation
    bool useMA = true;

    // Количество измерений в графике
    int N_pts = 256;

    bool useSkinDetection = true;

    if (argc > 1)
    {
        fileName = argv[1];
    }
    if (argc > 2)
    {
        N_pts = atoi(argv[2]);
    }
    if (argc > 3)
    {
        useMA = atoi(argv[3]) > 0;
    }
    if (argc > 4)
    {
        useSkinDetection = atoi(argv[4]) > 0;
    }

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
    std::cout << "Time frequency = " << Freq << std::endl;

	// Создаем окошко
    cv::namedWindow("output", cv::WINDOW_NORMAL);

	// Изображение для вывода графика
    int frameWidth = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    int frameHeight = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    cv::Mat I(std::min(100, frameHeight), std::min(640, frameWidth), CV_8UC3, cv::Scalar::all(0));

    cv::VideoWriter vw(fileName + "_" + argv[2] + "_" + argv[3] + "_result.avi", CV_FOURCC('X', '2', '6', '4'), fps, cv::Size(useMA ? (2 * frameWidth) : frameWidth, frameHeight), true);

	// Прямоугольник с лицом 
    cv::Rect currentRect;

    // Face detector and tracker
    FaceDetector faceDetector;
    SkinDetector skinDetector;
    if (useSkinDetection)
    {
        if (!skinDetector.Init("../HeartRateMeasure/data/"))
        {
            useSkinDetection = skinDetector.Learn("../HeartRateMeasure/data/");
            if (!useSkinDetection)
            {
                std::cout << "Scin detector wasn't initializad!" << std::endl;
            }
        }
    }
    LKTracker faceTracker;
    EulerianMA eulerianMA;

	// Создаем анализатор
    SignalProcessor sp(N_pts);

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

		// Детект лица
        cv::Rect face = faceDetector.detect_biggest_face(rgbframe, useSkinDetection);
        // Tracking
        if (face.area() > 0)
        {
            faceTracker.ReinitTracker(face);
            currentRect = faceTracker.GetTrackedRegion();
        }
        else
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

		// Если есть объект ненулевой площади вычисляем среднее по цвету
        if (currentRect.area() > 0)
		{
            cv::Mat skinMask = skinDetector.Detect(rgbframe(currentRect));
            cv::Scalar meanVal = cv::mean(frame(currentRect), skinMask.empty() ? cv::noArray() : skinMask);

            TimerTimestamp captureTime = useFPS ? ((frameInd * 1000.) / fps) : t1;
            std::cout << "Capture time = " << captureTime << std::endl;
            sp.AddMeasure(captureTime, cv::Vec3d(meanVal.val));
            sp.MeasureFrequency(I, Freq);
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

        if (useMA)
        {
            cv::Mat outImg(frame.rows, frame.cols * 2, CV_32FC3);
            cv::hconcat(rgbframe, frame, outImg);
            cv::imshow("output", outImg);

            if (vw.isOpened())
            {
                vw << outImg;
            }
        }
        else
        {
            cv::imshow("output", frame);

            if (vw.isOpened())
            {
                vw << frame;
            }
        }

        TimerTimestamp t2 = cv::getTickCount();

        double t = (t2 - t1) / tick_freq;
        std::cout << "t = " << t << std::endl;
        int waitTime = manual ? 0 : (std::max<int>(1, 1000 / fps - t / 1000));
        int k = cv::waitKey(waitTime);

        if (k == 'm' || k == 'M')
        {
            manual = !manual;
        }
        else
        {
            if (!manual && k > 0)
            {
                break;
            }
        }

        ++frameInd;
	}

    cv::waitKey(0);

    return 0;
}
