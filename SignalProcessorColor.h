#pragma once

#include "stat.h"

///
/// \brief The SignalProcessorColor class
/// Обработчик сигналов.
/// Включает в себя очередь (с итератором) с данными.
/// Функцию для размещения сигналов с метками на равномерной временной сетке.
/// Функцию для разделения сигналов.
///
class SignalProcessorColor
{
public:
    enum RGBFilters
    {
        FilterICA,
        FilterPCA
    };

    typedef cv::Vec3d ClVal_t;

    ///
    /// \brief SignalProcessor
    /// \param framesCount
    ///
    SignalProcessorColor(size_t framesCount, RGBFilters filterType,
                         float gauss_def_var, float gauss_min_var, float gauss_max_var,
                         float gauss_eps, float gauss_update_alpha,
                         float gauss_proc_alpha, float gauss_proc_weight_thresh);

    ///
    /// \brief Reset
    ///
    void Reset();

    ///
    /// \brief Добавление измерения
    /// \param captureTime
    /// \param val
    ///
    void AddMeasure(TimerTimestamp captureTime, const ClVal_t& val);

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
    void MeasureFrequency(cv::Mat& img, double Freq, int frameInd);

private:
    ///
    /// \brief m_size
    /// Кол-во элементов в очереди
    ///
    size_t m_size;

    ///
    /// \brief m_filterType
    ///
    RGBFilters m_filterType;

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
    std::deque<Measure<ClVal_t>> m_queue;

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
    ClVal_t FindValueForTime(TimerTimestamp _t);

    ///
    /// \brief Выделяем первый сигнал (первый собственный вектор)
    /// \param src
    /// \param dst
    ///
    void FilterRGBSignal(cv::Mat& src, cv::Mat& dst);
    ///
    /// \brief Unmix
    /// \param src
    /// \param dst
    ///
    void FilterRGBSignal(cv::Mat& src, std::vector<cv::Mat>& dst);

    ///
    /// \brief MakeFourier
    /// \param signal
    /// \param deltaTime
    /// \param currFreq
    /// \param minFreq
    /// \param maxFreq
    /// \param draw
    /// \param img
    ///
    void MakeFourier(cv::Mat& signal, double deltaTime, double& currFreq, double& minFreq, double& maxFreq, bool draw, cv::Mat img);

    ///
    /// \brief DrawSignal
    /// \param signal
    /// \param deltaTime
    ///
    void DrawSignal(const std::vector<cv::Mat>& signal, double deltaTime, bool saveResult, int frameInd);
};
