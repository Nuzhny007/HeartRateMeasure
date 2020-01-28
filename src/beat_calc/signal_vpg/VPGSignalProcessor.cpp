#include <cstring>
#include "VPGSignalProcessor.h"

///
VPGSignalProcessor::VPGSignalProcessor(size_t framesCount, float fps, bool doFilter)
	:
	m_minSignalSize(framesCount), m_fps(fps), m_doFilter(doFilter)
{
	double framePeriod = 1000. * framesCount / fps;
	m_pulseproc = std::make_unique<vpg::PulseProcessor>(framePeriod, 400., 350., 1000.f / fps, vpg::PulseProcessor::HeartRate);
	int totalcardiointervals = 25;
	m_peakdetector = std::make_unique<vpg::PeakDetector>(m_pulseproc->getLength(), totalcardiointervals, 11, framePeriod);
	m_pulseproc->setPeakDetector(m_peakdetector.get());
}

///
VPGSignalProcessor::~VPGSignalProcessor()
{

}

///
void VPGSignalProcessor::Reset()
{

}

///
void VPGSignalProcessor::AddMeasure(__int64 captureTime, const ClVal_t& val)
{
	if (m_prevTime && m_freq > 0)
	{
		m_pulseproc->update(val[1], 1000. * (captureTime - m_prevTime) / m_freq, m_doFilter);
	}
	m_prevTime = captureTime; // Время измерения
	++m_valuesRecieved;
}

///
/// \brief VPGSignalProcessor::GetFrequency
/// \param freqResults
///
void VPGSignalProcessor::GetFrequency(FrequencyResults* freqResults) const
{
	freqResults->minFreq = 55;
	freqResults->maxFreq = 175;
	freqResults->freq = m_pulseproc->getFrequency();
	freqResults->smootFreq = m_pulseproc->getFrequency();
	freqResults->snr = m_pulseproc->getSNR();
	freqResults->averageCardiointerval = m_peakdetector->averageCardiointervalms(9);
	freqResults->currentCardiointerval = m_peakdetector->getCurrentInterval();
}

///
int VPGSignalProcessor::RemainingMeasurements() const
{
	return (m_minSignalSize > m_valuesRecieved) ? static_cast<int>(m_minSignalSize - m_valuesRecieved) : 0;
}

///
/// Произвести вычисление пульса
/// Подразумевается, что при вызове AddMeasure никаких измерений производить не надо
int VPGSignalProcessor::MeasureFrequency(
	double freq,      /// Это частота, относительно которой можно вычислять промежуток времени межну значениями captureTime из AddMeasure: dt = (t2 - t1) / freq
	int /*frameInd*/,     /// Индекс кадра: используется в моей версии для визуализации гаусовских процессов
	bool /*showMixture*/  /// Отображать ли смесь Гауссианов: используется в моей версии
)
{
	m_freq = freq;

	if (m_minSignalSize < m_valuesRecieved)
	{
		m_pulseproc->computeFrequency();
		return 0;
	}
	else
	{
		return static_cast<int>(m_minSignalSize - m_valuesRecieved);
	}
}

///
/// Все сигналы и спектры какие есть надо хранить у себя и вернуть только указатели на них и размеры для отрисовки на кадре
void VPGSignalProcessor::GetSignal(SignalInfo* signalInfo)
{
	memset(signalInfo, 0, sizeof(SignalInfo));

	signalInfo->m_deltaTime = 1.f / m_fps;

	signalInfo->m_signal[0] = const_cast<double*>(m_pulseproc->getSignal());
	signalInfo->m_signalSize[0] = m_pulseproc->getLength();

	signalInfo->m_signal[1] = const_cast<double*>(m_peakdetector->getBinarySignal());
	signalInfo->m_signalSize[1] = m_pulseproc->getLength();

	signalInfo->m_signal[2] = const_cast<double*>(m_peakdetector->getIntervalsVector());
	signalInfo->m_signalSize[2] = m_peakdetector->getIntervalsLength();

	signalInfo->m_spectrum[0] = const_cast<double*>(m_pulseproc->getSpectr());
	signalInfo->m_spectrumSize[0] = m_pulseproc->getLength();

	//for (size_t i = 0; i < m_spectrumPower.size(); ++i)
	//{
	//	signalInfo->m_spectrum[i] = m_spectrumPower[i].ptr<double>(0);
	//	signalInfo->m_spectrumSize[i] = m_spectrumPower[i].cols;
	//}

	//for (size_t i = 0; i < m_freqValues.size(); ++i)
	//{
	//	signalInfo->m_freqValues[i] = &(m_freqValues[i])[0];
	//	signalInfo->m_valuesSize[i] = static_cast<int>(m_freqValues[i].size());
	//	signalInfo->m_fromInd[i] = m_fromToFreq[i].x;
	//	signalInfo->m_toInd[i] = m_fromToFreq[i].y;
	//}
}
