#pragma once

#include <opencv2/opencv.hpp>
typedef int64 TimerTimestamp;

///
/// \brief sqr
/// \param v
/// \return square of the value
///
template<typename T>
T sqr(T v)
{
    return v * v;
}

///
/// \brief The Measure class
/// Измерение.
/// Включает в себя момент времени в который произведено измерение и собствнно само измерение.
///
class Measure
{
public:
    TimerTimestamp t; // Время измерения
    cv::Vec3d val;		  // Значение измеренной величины

    Measure (TimerTimestamp t_, cv::Vec3d val_)
    {
        t = t_;
        val = val_;
    }
};

///
/// \brief The Gaussian class
///
template<typename MEAS_T, typename DATA_T>
class Gaussian
{
public:
    Gaussian(DATA_T eps = 2.7, DATA_T alpha = 0.9)
        :
          m_mean(0),
          m_var(0),
          m_eps(eps),
          m_alpha(alpha),
          m_lastMeasure(0),
          m_measuresCount(0)
    {
    }

    Gaussian(MEAS_T measure, DATA_T eps = 2.7, DATA_T alpha = 0.9)
        :
          m_mean(measure),
          m_var(measure),
          m_eps(eps),
          m_alpha(alpha),
          m_lastMeasure(measure),
          m_measuresCount(1)
    {
    }

    ///
    /// \brief AddMeasure
    /// \param measure
    /// \return
    ///
    bool AddMeasure(MEAS_T measure)
    {
        if (CheckMeasure(measure))
        {
            m_lastMeasure = measure;
            ++m_measuresCount;
            UpdateModel(measure);
            return true;
        }
        else
        {
            return false;
        }
    }

    MEAS_T CurrValue() const
    {
        return static_cast<MEAS_T>(m_mean);
    }

private:
    ///
    /// \brief m_mean
    /// Mean value of the Gaussian
    ///
    DATA_T m_mean;
    ///
    /// \brief m_var
    /// Variance
    ///
    DATA_T m_var;
    ///
    /// \brief m_eps
    /// Model accuracy:
    /// The maximum ratio of the deviation of the current measure from its average value to the mean square deviation
    /// The value 2.7 ~= 0.95 probablity
    ///
    DATA_T m_eps;

    ///
    /// \brief m_alpha
    /// Exponential smoothing coefficient
    ///
    DATA_T m_alpha;

    ///
    /// \brief m_lastMeasure
    /// Latest measure
    ///
    MEAS_T m_lastMeasure;
    size_t m_measuresCount;

    ///
    /// \brief CheckMeasure
    /// \param measure
    /// \return
    ///
    bool CheckMeasure(MEAS_T measure) const
    {
        return m_eps * m_var < std::abs(m_mean - measure);
    }
    ///
    /// \brief UpdateModel
    /// \param measure
    ///
    void UpdateModel(MEAS_T measure)
    {
        m_var = sqrt((1 - m_alpha) * sqr(m_var) + m_alpha * sqr(measure - m_mean));
        m_mean = (1 - m_alpha) * m_mean + m_alpha * measure;
    }
};

///
/// \brief The SignalProcessor class
/// Обработчик сигналов.
/// Включает в себя очередь (с итератором) с данными.
/// Функцию для размещения сигналов с метками на равномерной временной сетке.
/// Функцию для разделения сигналов.
///
class SignalProcessor
{
public:
    ///
    /// \brief SignalProcessor
    /// \param size
    ///
    SignalProcessor(size_t size);

    ///
    /// \brief Reset
    ///
    void Reset();

    ///
    /// \brief Добавление измерения
    /// \param captureTime
    /// \param val
    ///
    void AddMeasure(TimerTimestamp captureTime, cv::Vec3d val);

    ///
    /// \brief GetFreq
    /// \return
    ///
    double GetFreq() const;
    ///
    /// \brief GetInstantaneousFreq
    /// \param minFreq
    /// \param max_freq
    /// \return
    ///
    double GetInstantaneousFreq(double* minFreq, double* maxFreq) const;

    ///
    /// \brief Draw
    /// \param img
    /// \param Freq
    ///
    void Draw(cv::Mat& img, double Freq);

private:
    ///
    /// \brief m_size
    /// Кол-во элементов в очереди
    ///
    size_t m_size;

    ///
    /// \brief m_FF
    ///
    Gaussian<double, double> m_FF;
    ///
    /// \brief m_minFreq
    ///
    double m_minFreq;
    ///
    /// \brief m_maxFreq
    ///
    double m_maxFreq;
    ///
    /// \brief m_currFreq
    ///
    double m_currFreq;

    ///
    /// \brief m_queue
    ///
    std::deque<Measure> m_queue;

    ///
    /// \brief Преобразуем очередь измерений с метками времени в измерения на равномерной временной сетке
    /// \param NumSamples
    /// \param dst
    /// \param dt
    /// \param Freq
    ///
    void UniformTimedPoints(int NumSamples, cv::Mat& dst, double& dt, double Freq);

    ///
    /// \brief Вычисляем значение для произвольного момента времени, при помощи кусочно-линейной интерполяции по имеющимся в очереди элементам
    /// \param _t
    /// \return
    ///
    cv::Vec3d FindValueForTime(TimerTimestamp _t);

    ///
    /// \brief Выделяем первый сигнал (первый собственный вектор)
    /// \param src
    /// \param dst
    ///
    void Unmix(cv::Mat& src, cv::Mat& dst);
};
