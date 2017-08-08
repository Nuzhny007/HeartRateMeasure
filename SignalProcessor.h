#pragma once

#include <opencv2/opencv.hpp>

#include "stat.h"

typedef int64 TimerTimestamp;

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
    void MeasureFrequency(cv::Mat& img, double Freq);

private:
    ///
    /// \brief m_size
    /// Кол-во элементов в очереди
    ///
    size_t m_size;

    typedef GaussMixture<6, double, double> freq_t;
    ///
    /// \brief m_FF
    ///
    freq_t m_FF;
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
