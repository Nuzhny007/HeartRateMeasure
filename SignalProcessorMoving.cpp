#include "SignalProcessorMoving.h"

///
/// \brief SignalProcessorMoving::SignalProcessorMoving
/// \param framesCount
///
SignalProcessorMoving::SignalProcessorMoving(size_t framesCount,
                                             float gauss_def_var, float gauss_min_var, float gauss_max_var,
                                             float gauss_eps, float gauss_update_alpha,
                                             float gauss_proc_alpha, float gauss_proc_weight_thresh)
    :
      m_size(framesCount),
      m_FF(gauss_def_var, gauss_min_var, gauss_max_var, gauss_eps, gauss_update_alpha, gauss_proc_alpha, gauss_proc_weight_thresh),
      m_minFreq(0),
      m_maxFreq(0),
      m_currFreq(0)
{
}

///
/// \brief SignalProcessorMoving::Reset
///
void SignalProcessorMoving::Reset()
{
    m_queue.clear();
    m_FF.Reset();
    m_minFreq = 0;
    m_maxFreq = 0;
    m_currFreq = 0;
}

///
/// \brief SignalProcessorMoving::AddMeasure
/// \param captureTime
/// \param val
///
void SignalProcessorMoving::AddMeasure(TimerTimestamp captureTime, const MoveVal_t& val)
{
    m_queue.push_back(Measure<MoveVal_t>(captureTime, val));
    if (m_queue.size() > m_size)
    {
        m_queue.pop_front();
    }
}

///
/// \brief SignalProcessorMoving::GetFreq
/// \return
///
double SignalProcessorMoving::GetFreq() const
{
    return m_FF.CurrValue();
}

///
/// \brief SignalProcessorMoving::GetInstantaneousFreq
/// \param minFreq
/// \param maxFreq
/// \return
///
double SignalProcessorMoving::GetInstantaneousFreq(
        double* minFreq,
        double* maxFreq
        ) const
{
    if (minFreq)
    {
        *minFreq = m_minFreq;
    }
    if (maxFreq)
    {
        *maxFreq = m_maxFreq;
    }
    return m_currFreq;
}

///
/// \brief SignalProcessorMoving::FindValueForTime
/// \param _t
/// \return
///
SignalProcessorMoving::MoveVal_t SignalProcessorMoving::FindValueForTime(TimerTimestamp _t)
{
    if (m_queue.empty())
    {
        return 0;
    }

    auto it_prev = m_queue.begin();
    for (auto it = m_queue.begin(); it < m_queue.end(); ++it)
    {
        Measure<MoveVal_t> m = *it;
        if (m.t >= _t)
        {
            if (it_prev->t == it->t)
            {
                return it->val;
            }
            else
            {
                double dt = double(it->t - it_prev->t);
                MoveVal_t d_val = MoveVal_t(it->val - it_prev->val);
                double t_rel = _t - it_prev->t;
                MoveVal_t val_rel = d_val * (t_rel / dt);
                return val_rel + it_prev->val;
            }
        }
        it_prev = it;
    }
    assert(0);
    return MoveVal_t();
}

///
/// \brief SignalProcessorMoving::UniformTimedPoints
/// \param NumSamples
/// \param dst
/// \param dt
/// \param Freq
///
void SignalProcessorMoving::UniformTimedPoints(int NumSamples, cv::Mat& dst, double& dt, double Freq)
{
    if (dst.empty() ||
            dst.size() != cv::Size(1, NumSamples))
    {
        dst = cv::Mat(1, NumSamples, CV_64FC1);
    }

    dt = (m_queue.back().t - m_queue.front().t) / (double)NumSamples;
    TimerTimestamp T = m_queue.front().t;
    for (int i = 0; i < NumSamples; ++i)
    {
        T += dt;

        MoveVal_t val = FindValueForTime(T);
        dst.at<double>(0, i) = val;
    }
    dt /= Freq;
}

///
/// \brief SignalProcessorMoving::MakeFourier
/// \param signal
/// \param deltaTime
/// \param currFreq
/// \param minFreq
/// \param maxFreq
/// \param draw
/// \param img
///
void SignalProcessorMoving::MakeFourier(
        cv::Mat& signal,
        double deltaTime,
        double& currFreq,
        double& minFreq,
        double& maxFreq,
        bool draw,
        cv::Mat img
        )
{
    // Преобразование Фурье
    cv::Mat res = signal.clone();
    cv::Mat z = cv::Mat::zeros(1, signal.cols, CV_64FC1);
    std::vector<cv::Mat> ch;
    ch.push_back(res);
    ch.push_back(z);
    cv::merge(ch, res);

    cv::Mat res_freq;
    cv::dft(res, res_freq);
    cv::split(res_freq, ch);
    // Мощность спектра
    cv::magnitude(ch[0], ch[1], signal);
    // Квадрат мощности спектра
    cv::pow(signal, 2.0, signal);

    // Теперь частотный фильтр :)
    cv::line(signal, cv::Point(0, 0), cv::Point(15, 0), cv::Scalar::all(0), 1, CV_AA);
    cv::line(signal, cv::Point(100, 0), cv::Point(signal.cols - 1, 0), cv::Scalar::all(0), 1, CV_AA);

    // Чтобы все разместилось
    cv::normalize(signal, signal, 0, 1, cv::NORM_MINMAX);

    // Найдем 3 пика на частотном разложении
    const size_t INDS_COUNT = 3;
    int inds[INDS_COUNT] = { -1 };
    std::deque<double> maxVals;

    auto IsLocalMax = [](double v1, double v2, double v3) -> bool
    {
        return (v2 > v1) && (v2 > v3);
    };

    double v1 = signal.at<double>(0, 0);
    double v2 = signal.at<double>(0, 1);

    for (int x = 1; x < signal.cols - 1; ++x)
    {
        double v3 = signal.at<double>(0, x + 1);
        int ind = x;
        if (IsLocalMax(v1, v2, v3))
        {
            for (size_t i = 0; i < maxVals.size(); ++i)
            {
                if (maxVals[i] < v2)
                {
                    std::swap(maxVals[i], v2);
                    std::swap(inds[i], ind);
                }
            }
            if (maxVals.size() < INDS_COUNT)
            {
                maxVals.push_back(v2);
                inds[maxVals.size() - 1] = ind;
            }
        }
        v1 = v2;
        v2 = v3;
    }

    // И вычислим частоту
    maxFreq = 60.0 / (1 * deltaTime);
    minFreq = 60.0 / ((signal.cols - 1) * deltaTime);

    currFreq = -1;
    for (size_t i = 0; i < maxVals.size(); ++i)
    {
        if (inds[i] > 0)
        {
            double freq = 60.0 / (inds[i] * deltaTime);
            m_FF.AddMeasure(freq);

            if (currFreq < 0)
            {
                currFreq = freq;
                std::cout << "signal.size = " << signal.cols << ", maxInd = " << inds[i] << ", deltaTime = " << deltaTime << ", freq [" << minFreq << ", " << maxFreq << "] = " << currFreq << " - " << m_FF.CurrValue() << std::endl;
            }
        }
    }
    if (currFreq < 0)
    {
        currFreq = 0;
    }
    if (draw)
    {
        double scale_x = (double)img.cols / (double)m_queue.size();

        // Изобразим спектр Фурье
        float S = 50;
        for (int x = 1; x < signal.cols; ++x)
        {
            bool findInd = false;

            for (auto i : inds)
            {
                if (i == x)
                {
                    findInd = true;
                    break;
                }
            }

            cv::line(img,
                     cv::Point(scale_x * x, img.rows - S * signal.at<double>(x)),
                     cv::Point(scale_x * x, img.rows),
                     findInd ? cv::Scalar(255, 0, 255) : cv::Scalar(255, 255, 255));
        }

        std::vector<double> robustFreqs;
        m_FF.RobustValues(robustFreqs);

        std::cout << "Robust frequences: ";
        for (auto v : robustFreqs)
        {
            std::cout << v << " ";
        }
        std::cout << std::endl;
    }
}

///
/// \brief SignalProcessorMoving::MeasureFrequency
/// \param img
/// \param Freq
///
void SignalProcessorMoving::MeasureFrequency(cv::Mat& img, double Freq, int frameInd)
{
    if (m_queue.size() < m_size)
    {
        return;
    }
    img.setTo(0);

    cv::Mat src;

    // Чтобы частота сэмплирования не плавала,
    // разместим сигнал с временными метками на равномерной сетке.

    double dt;
    UniformTimedPoints(static_cast<int>(m_queue.size()), src, dt, Freq);

    DrawSignal(std::vector<cv::Mat>({ src }), dt, true, frameInd);

    MakeFourier(src, dt, m_currFreq, m_minFreq, m_maxFreq, true, img);

    m_FF.Visualize(true, frameInd, "moving");
}

///
/// \brief SignalProcessorMoving::DrawSignal
/// \param signal
/// \param deltaTime
///
void SignalProcessorMoving::DrawSignal(const std::vector<cv::Mat>& signal, double deltaTime, bool saveResult, int frameInd)
{
    const int wndHeight = 200;
    cv::Mat img(signal.size() * wndHeight, 512, CV_8UC3, cv::Scalar::all(255));

    for (size_t si = 0; si < signal.size(); ++si)
    {
        cv::Mat snorm;
        cv::normalize(signal[si], snorm, wndHeight, 0, cv::NORM_MINMAX);

        double timeSum = 0;
        double v0 = snorm.at<double>(0, 0);
        for (int i = 1; i < snorm.cols; ++i)
        {
            double v1 = snorm.at<double>(0, i);

            cv::Point pt0(((i - 1) * img.cols) / snorm.cols, (si + 1) * wndHeight - v0);
            cv::Point pt1((i * img.cols) / snorm.cols, (si + 1) * wndHeight - v1);

            cv::line(img, pt0, pt1, cv::Scalar(0, 0, 0));

            int dtPrev = static_cast<int>(1000. * timeSum) / 1000;
            timeSum += deltaTime;
            int dtCurr = static_cast<int>(1000. * timeSum) / 1000;
            if (dtCurr > dtPrev)
            {
                cv::line(img, cv::Point(pt1.x, si * wndHeight), cv::Point(pt1.x, (si + 1) * wndHeight - 1), cv::Scalar(0, 150, 0));
            }

            v0 = v1;
        }
        cv::line(img, cv::Point(0, (si + 1) * wndHeight), cv::Point(img.cols - 1, (si + 1) * wndHeight), cv::Scalar(0, 0, 0));
    }

    cv::namedWindow("signal moving", cv::WINDOW_AUTOSIZE);
    cv::imshow("signal moving", img);

    if (saveResult)
    {
        std::string fileName = "signal_moving/" + std::to_string(frameInd) + ".png";
        cv::imwrite(fileName, img);
    }
}
