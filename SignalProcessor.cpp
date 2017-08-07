#include "SignalProcessor.h"
#include "FastICA.h"

///
/// \brief SignalProcessor::SignalProcessor
/// \param s
///
SignalProcessor::SignalProcessor(size_t size)
    :
      m_size(size),
      m_minFreq(0),
      m_maxFreq(0),
      m_currFreq(0)
{
}

///
/// \brief SignalProcessor::Reset
///
void SignalProcessor::Reset()
{
    m_queue.clear();
    m_FF = Gaussian<double, double>();
    m_minFreq = 0;
    m_maxFreq = 0;
    m_currFreq = 0;
}

///
/// \brief SignalProcessor::AddMeasure
/// \param captureTime
/// \param val
///
void SignalProcessor::AddMeasure(TimerTimestamp captureTime, cv::Vec3d val)
{
    m_queue.push_back(Measure(captureTime, val));
    if (m_queue.size() > m_size)
    {
        m_queue.pop_front();
    }
}

///
/// \brief SignalProcessor::GetFreq
/// \return
///
double SignalProcessor::GetFreq() const
{
    return m_FF.CurrValue();
}

double SignalProcessor::GetInstantaneousFreq(
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
/// \brief SignalProcessor::FindValueForTime
/// \param _t
/// \return
///
cv::Vec3d SignalProcessor::FindValueForTime(TimerTimestamp _t)
{
    if (m_queue.empty())
    {
        return 0;
    }

    auto it_prev = m_queue.begin();
    for (auto it = m_queue.begin(); it < m_queue.end(); ++it)
    {
        Measure m = *it;
        if (m.t >= _t)
        {
            if (it_prev->t == it->t)
            {
                return it->val;
            }
            else
            {
                double dt = double(it->t - it_prev->t);
                cv::Vec3d d_val = cv::Vec3d(it->val - it_prev->val);
                double t_rel = _t - it_prev->t;
                cv::Vec3d val_rel = d_val * (t_rel / dt);
                return val_rel + it_prev->val;
            }
        }
        it_prev = it;
    }
    assert(0);
    return cv::Vec3d();
}

///
/// \brief SignalProcessor::UniformTimedPoints
/// \param NumSamples
/// \param dst
/// \param dt
/// \param Freq
///
void SignalProcessor::UniformTimedPoints(int NumSamples, cv::Mat& dst, double& dt, double Freq)
{
    if (dst.empty() ||
            dst.size() != cv::Size(3, NumSamples))
    {
        dst = cv::Mat(3, NumSamples, CV_64FC1);
    }

    std::vector<cv::Vec3d> res;
    dt = (m_queue.back().t - m_queue.front().t) / (double)NumSamples;
    TimerTimestamp T = m_queue.front().t;
    for (int i = 0; i < NumSamples; ++i)
    {
        T += dt;
        res.push_back(FindValueForTime(T));
    }
    dt /= Freq;

    for (int i = 0; i < NumSamples; ++i)
    {
        dst.at<double>(0, i) = res[i][0];
        dst.at<double>(1, i) = res[i][1];
        dst.at<double>(2, i) = res[i][2];
    }
}

///
/// \brief SignalProcessor::Unmix
/// \param src
/// \param dst
///
void SignalProcessor::Unmix(cv::Mat& src, cv::Mat& dst)
{
    cv::Mat W;
    cv::Mat d;
    int N = 0; // Номер независимой компоненты, используемой для измерения частоты
    FastICA fica;
    fica.apply(src, d, W); // Производим разделение компонентов
    d.row(N) *= (W.at<double>(N, N) > 0) ? 1 : -1; // Инверсия при отрицательном коэффициенте
    dst = d.row(N).clone();
    cv::normalize(dst, dst, 0, 1, cv::NORM_MINMAX);
}

///
/// \brief SignalProcessor::Draw
/// \param img
/// \param Freq
///
void SignalProcessor::Draw(cv::Mat& img, double Freq)
{
    if (m_queue.size() < m_size / 2)
    {
        return;
    }
    img = 0;

    double scale_x = (double)img.cols / (double)m_queue.size();

    cv::Mat src;
    cv::Mat dst;

    // Чтобы частота сэмплирования не плавала,
    // разместим сигнал с временными метками на равномерной сетке.

    double dt;
    UniformTimedPoints(static_cast<int>(m_queue.size()), src, dt, Freq);

    // Разделяем сигналы
    Unmix(src, dst);

    // Преобразование Фурье
    cv::Mat res = dst.clone();
    cv::Mat z = cv::Mat::zeros(1, dst.cols, CV_64FC1);
    std::vector<cv::Mat> ch;
    ch.push_back(res);
    ch.push_back(z);
    cv::merge(ch, res);

    cv::Mat res_freq;
    cv::dft(res, res_freq);
    cv::split(res_freq, ch);
    // Мощность спектра
    cv::magnitude(ch[0], ch[1], dst);
    // Квадрат мощности спектра
    cv::pow(dst, 2.0, dst);

    // Теперь частотный фильтр :)
    cv::line(dst, cv::Point(0, 0), cv::Point(15, 0), cv::Scalar::all(0), 1, CV_AA);
    cv::line(dst, cv::Point(100, 0), cv::Point(dst.cols - 1, 0), cv::Scalar::all(0), 1, CV_AA);

    // Чтобы все разместилось
    cv::normalize(dst, dst, 0,1, cv::NORM_MINMAX);

    // Найдем пик на частотном разложении
    cv::Point maxInd;
    double maxVal;
    cv::minMaxLoc(dst, nullptr, &maxVal, nullptr, &maxInd);

    // И вычислим частоту
    m_maxFreq = 60.0 / (1 * dt);
    m_minFreq = 60.0 / ((dst.cols - 1) * dt);
    if (maxInd.x > 0)
    {
        m_currFreq = 60.0 / (maxInd.x * dt);
        m_FF.AddMeasure(m_currFreq);

        std::cout << "dst.size = " << dst.cols << ", maxInd = " << maxInd.x << ", dt = " << dt << ", freq [" << m_minFreq << ", " << m_maxFreq << "] = " << m_currFreq << " - " << m_FF.CurrValue() << std::endl;
    }
    else
    {
        m_currFreq = 0;
    }

    // Изобразим спектр Фурье
    float S = 50;
    for (int x = 1; x < dst.cols; ++x)
    {
        cv::line(img,
                 cv::Point(scale_x * x, img.rows - S * dst.at<double>(x)),
                 cv::Point(scale_x * x, img.rows),
                 (x == maxInd.x) ? cv::Scalar(255, 0, 255) : cv::Scalar(255, 255, 255));
    }
}
