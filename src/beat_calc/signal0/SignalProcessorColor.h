#pragma once

#include "../../common/common.h"
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
    typedef cv::Vec3d ClVal_t;

    ///
    /// \brief SignalProcessor
    /// \param framesCount
    ///
    SignalProcessorColor(size_t framesCount,
		                 MeasureSettings::RGBFilters filterType, bool signalNormalization,
                         float gauss_def_var, float gauss_min_var, float gauss_max_var,
                         float gauss_eps, float gauss_update_alpha,
                         float gauss_proc_alpha, float gauss_proc_weight_thresh,
		                 bool retExpFreq);

    ///
    /// \brief Reset
    ///
    void Reset();

    ///
    /// \brief Добавление измерения
    /// \param captureTime
    /// \param val
    ///
    void AddMeasure(int64 captureTime, const ClVal_t& val);

    ///
    /// \brief GetInstantaneousFreq
    /// \param freqResults
    ///
    void GetFrequency(FrequencyResults* freqResults) const;

	///
	/// \brief RemainingMeasurements
	/// \return
	///
	int RemainingMeasurements() const;

    ///
    /// \brief Draw
    /// \param img
    /// \param Freq
    ///
    int MeasureFrequency(double freq, int frameInd, bool showMixture);

	///
	/// \brief SaveColorsToFile
	/// \param fileName
	///
	bool SaveColorsToFile(const std::string& fileName);

	///
	/// \brief GetSignal
	///
	void GetSignal(SignalInfo* signalInfo);

private:
    ///
    /// \brief m_size
    /// Кол-во элементов в очереди
    ///
    size_t m_minSignalSize = 0;

    ///
    /// \brief m_filterType
    ///
	MeasureSettings::RGBFilters m_filterType;

	///
	/// \brief m_signalNormalization
	///
	bool m_signalNormalization = true;

    typedef GaussMixture<6, double, double> freq_t;
    ///
    /// \brief m_FF
    ///
    freq_t m_FF;
    ///
    /// \brief m_minFreq
    ///
    double m_minFreq = 0;
    ///
    /// \brief m_maxFreq
    ///
    double m_maxFreq = 0;
    ///
    /// \brief m_currFreq
    ///
    double m_currFreq = 0;
	///
	/// \brief m_expFreq
	///
	double m_expFreq = 0;
	///
	/// \brief m_retExpFreq
	///
	bool m_retExpFreq = false;

    ///
    /// \brief m_queue
    ///
    std::deque<Measure<ClVal_t>> m_queue;

	///
	/// \brief Signal after correction and filtration
	///
	std::vector<cv::Mat> m_correctedSignal;

	///
	/// \brief 
	///
	std::vector<cv::Mat> m_spectrumPower;
	std::vector<std::vector<int>> m_freqValues;
	std::vector<cv::Point> m_fromToFreq;

	///
	/// \brief Delta time for the latest analisys
	///
	double m_lastDeltatime = 0.0;

	///
	/// \brief m_colorsLog
	///
	std::ofstream m_colorsLog;

    ///
    /// \brief Преобразуем очередь измерений с метками времени в измерения на равномерной временной сетке
    /// \param NumSamples
    /// \param dst
    /// \param dt
    /// \param Freq
    ///
    void UniformTimedPoints(const std::deque<Measure<ClVal_t>>& queue, cv::Mat& dst, double& dt, double Freq);

    ///
    /// \brief Вычисляем значение для произвольного момента времени, при помощи кусочно-линейной интерполяции по имеющимся в очереди элементам
    /// \param _t
    /// \return
    ///
    ClVal_t FindValueForTime(const std::deque<Measure<ClVal_t>>& queue, int64 _t);

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
    void MakeFourier(cv::Mat& signal, cv::Mat& spectrum, std::vector<int>& freqValues, cv::Point& fromToFreq, double deltaTime, double& currFreq, double& minFreq, double& maxFreq);
};
