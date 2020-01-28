#pragma once
#include "../plugin.h"
#include "hrvprocessor.h"
#include "peakdetector.h"
#include "pulseprocessor.h"

///
class VPGSignalProcessor
{
public:
	typedef cv::Vec3d ClVal_t;

	///
	VPGSignalProcessor(size_t framesCount, float fps, bool doFilter);

	///
	~VPGSignalProcessor();

	///
	void Reset();

	///
	void AddMeasure(__int64 captureTime, const ClVal_t& val);

	///
	/// \brief GetInstantaneousFreq
	/// \param freqResults
	///
	void GetFrequency(FrequencyResults* freqResults) const;

	///
	int RemainingMeasurements() const;

	///
	int MeasureFrequency(double freq, int frameInd, bool showMixture);

	///
	void GetSignal(SignalInfo* signalInfo);

private:
	///
	/// \brief m_size
	/// Кол-во элементов в очереди
	///
	size_t m_minSignalSize = 0;
	size_t m_valuesRecieved = 0;

	float m_fps = 0;

	__int64 m_prevTime = 0;
	double m_freq = 0.;

	bool m_doFilter = false;

	// Instance of PulseProcessor (it analyzes counts of skin reflection and computes heart rate by means on FFT analysis)
	std::unique_ptr<vpg::PulseProcessor> m_pulseproc;
	
	// Peak detector for the cardio intervals evaluation and analysis (it analyzes cardio intervals)
	std::unique_ptr<vpg::PeakDetector> m_peakdetector;
	
	// HRVProcessor for HRV analysis
	vpg::HRVProcessor m_hrvproc;
};
