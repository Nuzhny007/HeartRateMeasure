#include "SignalProcessorColor.h"
#include "FastICA.h"
#include "pca.h"

///
/// \brief SignalProcessorColor::SignalProcessorColor
/// \param framesCount
///
SignalProcessorColor::SignalProcessorColor(size_t framesCount,
	                                       MeasureSettings::RGBFilters filterType, bool signalNormalization,
                                           float gauss_def_var, float gauss_min_var, float gauss_max_var,
                                           float gauss_eps, float gauss_update_alpha,
                                           float gauss_proc_alpha, float gauss_proc_weight_thresh,
	                                       bool retExpFreq)
    :
      m_minSignalSize(framesCount),
      m_filterType(filterType),
	  m_signalNormalization(signalNormalization),
      m_FF(gauss_def_var, gauss_min_var, gauss_max_var, gauss_eps, gauss_update_alpha, gauss_proc_alpha, gauss_proc_weight_thresh),
      m_minFreq(0),
      m_maxFreq(0),
      m_currFreq(0),
	  m_expFreq(0),
	  m_retExpFreq(retExpFreq)
{
}

///
/// \brief SignalProcessorColor::Reset
///
void SignalProcessorColor::Reset()
{
    m_queue.clear();
    m_FF.Reset();
    m_minFreq = 0;
    m_maxFreq = 0;
    m_currFreq = 0;
	m_expFreq = 0;
}

///
/// \brief SignalProcessorColor::AddMeasure
/// \param captureTime
/// \param val
///
void SignalProcessorColor::AddMeasure(int64 captureTime, const ClVal_t& val)
{
    m_queue.push_back(Measure<ClVal_t>(captureTime, val));
    if (m_queue.size() > /*2 **/m_minSignalSize)
    {
        m_queue.pop_front();
    }

	if (m_colorsLog.is_open())
	{
		m_colorsLog << val[0] << "; " << val[1] << "; " << val[2] << std::endl;
	}
}

///
/// \brief SignalProcessorColor::RemainingMeasurements
/// \return
///
int SignalProcessorColor::RemainingMeasurements() const
{
	return (m_minSignalSize > m_queue.size()) ? int(m_minSignalSize - m_queue.size()) : 0;
}

///
/// \brief SignalProcessorColor::GetFrequency
/// \param freqResults
///
void SignalProcessorColor::GetFrequency(FrequencyResults* freqResults) const
{
	freqResults->minFreq = m_minFreq;
	freqResults->maxFreq = m_maxFreq;
	freqResults->freq = m_currFreq;
	if (m_retExpFreq)
	{
		freqResults->smootFreq = m_expFreq;
	}
	else
	{
		freqResults->smootFreq = m_FF.CurrValue();
	}
	freqResults->snr = 0;
}

///
/// \brief SignalProcessorColor::FindValueForTime
/// \param _t
/// \return
///
SignalProcessorColor::ClVal_t SignalProcessorColor::FindValueForTime(const std::deque<Measure<ClVal_t>>& queue, int64 _t)
{
    auto it_prev = queue.begin();
    for (auto it = queue.begin(); it < queue.end(); ++it)
    {
        if (it->t >= _t)
        {
            if (it_prev->t == it->t)
            {
                return it->val;
            }
            else
            {
                double dt = double(it->t - it_prev->t);
                ClVal_t d_val = ClVal_t(it->val - it_prev->val);
                double t_rel = _t - it_prev->t;
                ClVal_t val_rel = d_val * (t_rel / dt);
                return val_rel + it_prev->val;
            }
        }
        it_prev = it;
    }
    assert(0);
    return ClVal_t();
}

///
/// \brief SignalProcessorColor::UniformTimedPoints
/// \param NumSamples
/// \param dst
/// \param dt
/// \param Freq
///
void SignalProcessorColor::UniformTimedPoints(const std::deque<Measure<ClVal_t>>& queue, cv::Mat& dst, double& dt, double Freq)
{
	int NumSamples = static_cast<int>(queue.size());
    if (dst.empty() ||
            dst.size() != cv::Size(3, NumSamples))
    {
        dst = cv::Mat(3, NumSamples, CV_64FC1);
    }

    dt = (queue.back().t - queue.front().t) / (double)NumSamples;
    int64 T = queue.front().t;
    for (int i = 0; i < NumSamples; ++i)
    {
        T += dt;

        ClVal_t val = FindValueForTime(queue, T);
        dst.at<double>(0, i) = val[0];
        dst.at<double>(1, i) = val[1];
        dst.at<double>(2, i) = val[2];
    }
    dt /= Freq;
}

///
/// \brief SignalProcessorColor::FilterRGBSignal
/// \param src
/// \param dst
/// \param filterType
///
void SignalProcessorColor::FilterRGBSignal(cv::Mat& src, cv::Mat& dst)
{
    switch (m_filterType)
    {
	case MeasureSettings::FilterICA:
    {
        cv::Mat W;
        cv::Mat d;
        int N = 0; // Номер независимой компоненты, используемой для измерения частоты
        FastICA fica;
        fica.apply(src, d, W); // Производим разделение компонентов
        d.row(N) *= (W.at<double>(N, N) > 0) ? 1 : -1; // Инверсия при отрицательном коэффициенте
        dst = d.row(N).clone();
    }
        break;

	case MeasureSettings::FilterPCA:
        MakePCA(src, dst);
        break;

	case MeasureSettings::FilterGreen:
	{
		dst.create(1, src.cols, CV_64FC1);
		for (int idx = 0; idx < src.cols; ++idx)
		{
			dst.at<double>(0, idx) = src.at<double>(1, idx);
		}
		break;
	}
    }
    cv::normalize(dst, dst, 0, 1, cv::NORM_MINMAX);
}

///
/// \brief SignalProcessorColor::FilterRGBSignal
/// \param src
/// \param dst
/// \param filterType
///
void SignalProcessorColor::FilterRGBSignal(cv::Mat& src, std::vector<cv::Mat>& dst)
{
    switch (m_filterType)
    {
    case MeasureSettings::FilterICA:
    {
        cv::Mat W;
        cv::Mat d;
        FastICA fica;
        fica.apply(src, d, W); // Производим разделение компонентов

        dst.resize(d.rows);
        for (int i = 0; i < d.rows; ++i)
        {
            d.row(i) *= (W.at<double>(i, i) > 0) ? 1 : -1; // Инверсия при отрицательном коэффициенте
            dst[i] = d.row(i).clone();
            cv::normalize(dst[i], dst[i], 0, 1, cv::NORM_MINMAX);
        }
    }
        break;

    case MeasureSettings::FilterPCA:
        break;

	case MeasureSettings::FilterGreen:
	{
		dst.resize(src.rows);
		for (int i = 0; i < src.rows; ++i)
		{
			dst[i].create(1, src.cols, CV_64FC1);
			for (int idx = 0; idx < src.cols; ++idx)
			{
				dst[i].at<double>(0, idx) = src.at<double>(i, idx);
			}
		}
		break;
	}
    }
}

///
void normalization(cv::InputArray _a, cv::OutputArray _b)
{
	_a.getMat().copyTo(_b);
	cv::Mat b = _b.getMat();
	cv::Scalar mean;
	cv::Scalar stdDev;
	cv::meanStdDev(b, mean, stdDev);
	b = (b - mean[0]) / stdDev[0];
}

///
void meanFilter(cv::InputArray _a, cv::OutputArray _b, size_t n, cv::Size s)
{
	_a.getMat().copyTo(_b);
	cv::Mat b = _b.getMat();
	for (size_t i = 0; i < n; i++)
	{
		cv::blur(b, b, s);
	}
}

///
template<typename T>
void detrend(cv::Mat _z, cv::Mat& _r, int lambda = 10) 
{
	CV_DbgAssert((_z.type() == CV_32F || _z.type() == CV_64F)
		&& _z.total() == std::max(_z.size().width, _z.size().height));

	cv::Mat z = _z.total() == (size_t)_z.size().height ? _z : _z.t();
	if (z.total() < 3)
	{
		z.copyTo(_r);
	}
	else
	{
		int t = static_cast<int>(z.total());
		cv::Mat i = cv::Mat::eye(t, t, z.type());
		cv::Mat d = cv::Mat(cv::Matx<T, 1, 3>(1, -2, 1));
		cv::Mat d2Aux = cv::Mat::ones(t - 2, 1, z.type()) * d;
		cv::Mat d2 = cv::Mat::zeros(t - 2, t, z.type());
		for (int k = 0; k < 3; k++)
		{
			d2Aux.col(k).copyTo(d2.diag(k));
		}
		cv::Mat r = (i - (i + lambda * lambda * d2.t() * d2).inv()) * z;
		//r.copyTo(_r);
		
		_r = r.reshape(1, 1);
	}
}

///
/// \brief SignalProC:/work/vitagraph/beatmagnifiercessorColor::MakeFourier
/// \param signal
/// \param deltaTime
/// \param currFreq
/// \param minFreq
/// \param maxFreq
/// \param draw
/// \param img
///
void SignalProcessorColor::MakeFourier(
        cv::Mat& signal,
	    cv::Mat& spectrum,
	    std::vector<int>& freqValues,
	    cv::Point& fromToFreq,
        double deltaTime,
        double& currFreq,
        double& minFreq,
        double& maxFreq)
{
    // Преобразование Фурье
	cv::Mat res;
	if (m_signalNormalization)
	{
		// process raw signal
		detrend<double>(signal, res, cvRound(1000 / (2 * deltaTime)));
		normalization(res, res);
		meanFilter(res, res, 3, cv::Size(5, 5));
	}
	else
	{
		res = signal.clone();
	}

    cv::Mat z = cv::Mat::zeros(1, signal.cols, CV_64FC1);
    std::vector<cv::Mat> ch;
    ch.push_back(res);
    ch.push_back(z);
    cv::merge(ch, res);

    cv::Mat res_freq;
    cv::dft(res, res_freq);
    cv::split(res_freq, ch);
    // Мощность спектра
    cv::magnitude(ch[0], ch[1], spectrum);
    // Квадрат мощности спектра
    cv::pow(spectrum, 2.0, spectrum);

#if 1
#if 1
	const double total = signal.cols;
	auto Ind2Freq = [&](int ind) -> double
	{
		return (ind * 1000. * deltaTime * 60.0) / (2. * total);
	};
	auto Freq2Ind = [&](double bpm) -> int
	{
		return cvRound((2.0 * bpm * total) / (60. * 1000. * deltaTime));
	};
#else
	auto Ind2Freq = [&](int ind) -> double
	{
		return (60.0) / (ind * deltaTime);
	};
	auto Freq2Ind = [&](double bpm) -> int
	{
		return cvRound((60.0) / (bpm * deltaTime));
	};
#endif
#else
	// Функция подсчёта энергетического спектра - это быстрое преобразование Фурье.
	// Для него встроенная функция обязана быть. Нужно разобраться с тем,
	// как индексы массива с результатом преобразования Фурье мэпятся в частоту.
	// А делают они это примерно так : f[p] = p * (2 * pi / T).
	// Где T - полное время, за которое выполнено преобразование.
	// "Примерно" из - за того, что в первой половине массива частоты положительные, во второй - отрицательные

	const double total = signal.cols;
	auto Ind2Freq = [&](int ind) -> double
	{
		return (2.0 * M_PI * ind * 60.0) / (total);
	};
	auto Freq2Ind = [&](double bpm) -> int
	{
		return cvRound((bpm * total) / (60. * 2.0 * M_PI));
	};
#endif

    // Теперь частотный фильтр
	const double minBpm = 40.0;
	const double maxBpm = 200.0;
	
	fromToFreq.x = Freq2Ind(minBpm);
	fromToFreq.y = Freq2Ind(maxBpm);
	if (fromToFreq.x > fromToFreq.y)
	{
		std::swap(fromToFreq.x, fromToFreq.y);
	}

	double minS = 0;
	double maxS = 0;
	cv::Point minI;
	cv::Point maxI;
	cv::minMaxLoc(spectrum, &minS, &maxS, &minI, &maxI);
	std::cout << "ff1 = " << fromToFreq.x << ", ff2 = " << fromToFreq.y << std::endl;
	std::cout << "spectrum: " << spectrum.size() << ", min = " << minI << " - " << minS << ", max = " << maxI << " - " << maxS << std::endl;
	//std::cout << signal << std::endl;
	spectrum(cv::Rect(0, 0, fromToFreq.x, 1)).setTo(0);
	spectrum(cv::Rect(fromToFreq.y, 0, spectrum.cols - fromToFreq.y, 1)).setTo(0);

    // Чтобы все разместилось
    cv::normalize(spectrum, spectrum, 0, 1, cv::NORM_MINMAX);

    // Найдем 3 пика на частотном разложении
    const size_t INDS_COUNT = 3;
    int inds[INDS_COUNT] = { -1 };
    std::deque<double> maxVals;

    auto IsLocalMax = [](double v1, double v2, double v3) -> bool
    {
        return (v2 > v1) && (v2 > v3);
    };

    double v1 = spectrum.at<double>(0, 0);
    double v2 = spectrum.at<double>(0, 1);

    for (int x = 1; x < spectrum.cols - 1; ++x)
    {
        double v3 = spectrum.at<double>(0, x + 1);
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
	maxFreq = Ind2Freq(1);
	minFreq = Ind2Freq(spectrum.cols - 1);

	currFreq = -1;
    for (size_t i = 0; i < maxVals.size(); ++i)
    {
        if (inds[i] > 0)
        {
            double freq = Ind2Freq(inds[i]);
            m_FF.AddMeasure(freq);

            if (currFreq < 0)
            {
                currFreq = freq;
                std::cout << "spectrum.size = " << spectrum.cols << ", maxInd = " << inds[i] << ", deltaTime = " << deltaTime << ", freq [" << minFreq << ", " << maxFreq << "] = " << currFreq << " - " << m_FF.CurrValue() << std::endl;
            }
        }
    }
    if (currFreq < 0)
    {
        currFreq = 0;
    }

	freqValues.clear();
	for (int x = fromToFreq.x; x <= fromToFreq.y; ++x)
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

		freqValues.push_back(Ind2Freq(x));
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

///
/// \brief SignalProcessorColor::MeasureFrequency
/// \param img
/// \param Freq
///
int SignalProcessorColor::MeasureFrequency(double freq,
	int frameInd,
	bool showMixture)
{
    if (m_queue.size() < m_minSignalSize)
    {
        return int(m_minSignalSize - m_queue.size());
    }

	std::deque<Measure<ClVal_t>> workQueue;
#if 0
	auto IsPowerOfTwo = [](size_t v) -> bool
	{
		return (v && !(v & (v - 1)));
	};
	if (m_queue.size() >= m_minSignalSize && IsPowerOfTwo(m_queue.size()))
	{
		workQueue.assign(m_queue.begin(), m_queue.end());
	}
	else
	{
		workQueue.assign(m_queue.begin() + (m_queue.size() - m_minSignalSize), m_queue.end());
	}
	std::cout << "queue size = " << m_queue.size() << ", workQueue size = " << workQueue.size() << std::endl;
#else
	workQueue.assign(m_queue.begin(), m_queue.end());
#endif

    // Чтобы частота сэмплирования не плавала, разместим сигнал с временными метками на равномерной сетке
	cv::Mat src;
	m_lastDeltatime = 0;
    UniformTimedPoints(workQueue, src, m_lastDeltatime, freq);

    switch (m_filterType)
    {
	case MeasureSettings::FilterPCA:
	case MeasureSettings::FilterGreen:
	{
		cv::Mat dst;
		FilterRGBSignal(src, dst);

		m_correctedSignal.resize(1);
		m_correctedSignal[0] = dst;
		m_spectrumPower.resize(1);
		m_freqValues.resize(1);
		m_fromToFreq.resize(1);
		MakeFourier(dst, m_spectrumPower[0], m_freqValues[0], m_fromToFreq[0], m_lastDeltatime, m_currFreq, m_minFreq, m_maxFreq);
	}
	break;

    case MeasureSettings::FilterICA:
    {
        // Разделяем сигналы
        FilterRGBSignal(src, m_correctedSignal);
		m_spectrumPower.resize(m_correctedSignal.size());
		m_freqValues.resize(m_correctedSignal.size());
		m_fromToFreq.resize(m_correctedSignal.size());
		for (size_t di = 0; di < m_correctedSignal.size(); ++di)
        {
            auto& dst = m_correctedSignal[di];

            double currFreq = 0;
            double minFreq = 0;
            double maxFreq = 0;
            MakeFourier(dst, m_spectrumPower[di], m_freqValues[di], m_fromToFreq[di], m_lastDeltatime, currFreq, minFreq, maxFreq);

            if (di == 0)
            {
                m_currFreq = currFreq;
                m_maxFreq = maxFreq;
                m_minFreq = minFreq;
            }
        }
    }
        break;
    }

	const double expAlpha = 0.7;
	m_expFreq = expAlpha * m_expFreq + (1.0 - expAlpha) * m_currFreq;

	if (showMixture)
	{
		m_FF.Visualize(true, frameInd, "color");
	}

	return 0;
}

///
/// \brief SignalProcessorColor::SaveColorsToFile
/// \param fileName
///
bool SignalProcessorColor::SaveColorsToFile(const std::string& fileName)
{
	if (!m_colorsLog.is_open())
	{
		m_colorsLog.open(fileName);
	}

	return m_colorsLog.is_open();
}

///
/// \brief SignalProcessorColor::GetSignal
///
void SignalProcessorColor::GetSignal(SignalInfo* signalInfo)
{
	memset(signalInfo, 0, sizeof(SignalInfo));

	signalInfo->m_deltaTime = m_lastDeltatime;

	for (size_t i = 0; i < m_correctedSignal.size(); ++i)
	{
		signalInfo->m_signal[i] = m_correctedSignal[i].ptr<double>(0);
		signalInfo->m_signalSize[i] = m_correctedSignal[i].cols;
	}

	for (size_t i = 0; i < m_spectrumPower.size(); ++i)
	{
		signalInfo->m_spectrum[i] = m_spectrumPower[i].ptr<double>(0);
		signalInfo->m_spectrumSize[i] = m_spectrumPower[i].cols;
	}

	for (size_t i = 0; i < m_freqValues.size(); ++i)
	{
		signalInfo->m_freqValues[i] = &(m_freqValues[i])[0];
		signalInfo->m_valuesSize[i] = static_cast<int>(m_freqValues[i].size());
		signalInfo->m_fromInd[i] = m_fromToFreq[i].x;
		signalInfo->m_toInd[i] = m_fromToFreq[i].y;
	}
}
