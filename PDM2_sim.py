import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import soundfile as sf
from scipy.signal import lfilter, butter
from IPython.display import Audio
from photodiode_presets import PHOTODIODE_PRESETS
from LiFi_sim import Transmitter, Receiver
import os

# Параметры симуляции
fs = 1000000  # Частота дискретизации в Гц (1 МГц)
duration = 0.05  # Длительность сигнала в секундах
pulse_duration = 3e-9  # Длительность импульса в секундах (3 нс)
max_audio_freq = 20000  # Максимальная частота звука в Гц (20 кГц)
oversampling_ratio = fs // (2 * max_audio_freq)  # Коэффициент передискретизации

# Параметры оптического фильтра 905 нм - SLB905
OPTICAL_FILTER_CENTER = 905e-9  # Центральная длина волны фильтра (905 нм)
OPTICAL_FILTER_BANDWIDTH = 100e-9  # Полоса пропускания фильтра FWHM (100 нм)
OPTICAL_FILTER_TRANSMISSION = 0.9  # Пропускание на центральной длине волны (90%)
OPTICAL_FILTER_OD = 4  # Оптическая плотность вне полосы пропускания (OD4 = 0.0001)

# Функция создания директорий для результатов
def create_results_folders():
    # Создаем директории для графиков и результатов используя относительные пути
    plots_dir = "./plots"
    results_dir = "./results"
    
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    return plots_dir, results_dir

# Добавьте в конец программы после вывода информации о системе
plots_dir, results_dir = create_results_folders()

# Функция для генерации тестового аудио сигнала (сумма синусоид разных частот)
def generate_audio_signal(t):
    # Создаем сигнал с несколькими частотами
    f1, f2, f3 = 1000, 5000, 15000  # частоты в Гц
    signal = 0.5 * np.sin(2 * np.pi * f1 * t) + 0.3 * np.sin(2 * np.pi * f2 * t) + 0.2 * np.sin(2 * np.pi * f3 * t)
    return signal

# Функция для PDM кодирования (базовый алгоритм)
def pdm_encode(audio_signal):
    # Инициализируем переменные
    pdm_output = np.zeros_like(audio_signal)
    integrator = 0
    
    # Процесс PDM кодирования
    for i in range(len(audio_signal)):
        # Сравниваем входной сигнал с интегратором
        if audio_signal[i] > integrator:
            pdm_output[i] = 1
        else:
            pdm_output[i] = 0
        
        # Обновляем интегратор
        integrator = integrator + (pdm_output[i] - audio_signal[i]) * 0.1
    
    return pdm_output

# Функция для моделирования передачи через оптический канал (импульсы 3 нс)
def transmit_optical(pdm_signal, t):
    # Размер одного бита в наших отсчетах
    bit_samples = int(pulse_duration * fs)
    
    # Если bit_samples получается меньше 1, устанавливаем минимум 1 отсчет
    bit_samples = max(1, bit_samples)
    
    # Создаем массив для оптического сигнала
    optical_signal = np.zeros_like(pdm_signal)
    
    # Для каждого бита в PDM сигнале
    for i in range(len(pdm_signal)):
        if pdm_signal[i] == 1:
            # Создаем короткий импульс (3 нс)
            start_idx = i
            end_idx = min(i + bit_samples, len(optical_signal))
            optical_signal[start_idx:end_idx] = 1
    
    return optical_signal

# Функция для моделирования фотоприемника с компаратором
def photodetector_comparator(optical_signal, threshold=0.5):
    # Компаратор сравнивает входной сигнал с порогом
    return (optical_signal > threshold).astype(float)

# Функция для RC фильтра
def rc_filter(signal_in, cutoff_freq, fs):
    # Создаем RC фильтр нижних частот
    tau = 1 / (2 * np.pi * cutoff_freq)  # Постоянная времени
    alpha = fs * tau / (1 + fs * tau)  # Коэффициент фильтра
    
    # Применяем фильтр
    filtered_signal = np.zeros_like(signal_in)
    filtered_signal[0] = signal_in[0]
    
    for i in range(1, len(signal_in)):
        filtered_signal[i] = alpha * filtered_signal[i-1] + (1 - alpha) * signal_in[i]
    
    return filtered_signal

# Генерируем временную ось
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# Генерируем исходный аудио сигнал
audio_signal = generate_audio_signal(t)

# Нормализуем сигнал
audio_signal = audio_signal / np.max(np.abs(audio_signal))

# PDM кодирование
pdm_signal = pdm_encode(audio_signal)

# Передача по оптическому каналу (импульсы 3 нс)
optical_signal = transmit_optical(pdm_signal, t)

# Фотоприемник с компаратором
received_signal = photodetector_comparator(optical_signal)

# RC фильтр для восстановления аудио
cutoff_freq = 20000  # Частота среза фильтра (20 кГц)

# RC фильтр
filtered_signal_rc = rc_filter(received_signal, cutoff_freq, fs)

# фильтр Баттерворта
order = 2  # Порядок фильтра
nyquist = 0.5 * fs
normal_cutoff = cutoff_freq / nyquist
b, a = butter(order, normal_cutoff, btype='low', analog=False)
filtered_signal = lfilter(b, a, received_signal)

# Удаляем постоянную составляющую (DC offset)
filtered_signal = filtered_signal - np.mean(filtered_signal)

# Нормализуем восстановленный сигнал
if np.max(np.abs(filtered_signal)) > 0:
    filtered_signal = filtered_signal / np.max(np.abs(filtered_signal))

# Построение графиков
plt.figure(figsize=(14, 10))

# 1. Исходный аудио сигнал
plt.subplot(5, 1, 1)
plt.plot(t, audio_signal)
plt.title('Исходный аудио сигнал')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.grid(True)

# 2. PDM сигнал
plt.subplot(5, 1, 2)
# Показываем только часть сигнала для наглядности
show_samples = 1000
if len(t) > show_samples:
    plt.plot(t[:show_samples], pdm_signal[:show_samples], 'r-')
else:
    plt.plot(t, pdm_signal, 'r-')
plt.title('PDM сигнал (первые 1000 отсчетов)')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.grid(True)

# 3. Оптический сигнал (импульсы 3 нс)
plt.subplot(5, 1, 3)
# Показываем только часть сигнала для наглядности
if len(t) > show_samples:
    plt.plot(t[:show_samples], optical_signal[:show_samples], 'g-')
else:
    plt.plot(t, optical_signal, 'g-')
plt.title('Оптический сигнал (импульсы 3 нс)')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.grid(True)

# 4. Принятый сигнал после компаратора
plt.subplot(5, 1, 4)
# Показываем только часть сигнала для наглядности
if len(t) > show_samples:
    plt.plot(t[:show_samples], received_signal[:show_samples], 'b-')
else:
    plt.plot(t, received_signal, 'b-')
plt.title('Принятый сигнал после компаратора')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.grid(True)

# 5. Восстановленный сигнал после RC фильтра
plt.subplot(5, 1, 5)
plt.plot(t, filtered_signal)
plt.title('Восстановленный сигнал после RC фильтра')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.grid(True)

plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(plots_dir, "basic_pdm_signals.png"), dpi=150)
plt.close()  # Закрываем текущее окно графика, чтобы избежать наложения

# Анализ спектра
plt.figure(figsize=(14, 8))

# Спектр исходного сигнала
plt.subplot(2, 1, 1)
f, Pxx_orig = signal.welch(audio_signal, fs, nperseg=1024)
plt.semilogy(f, Pxx_orig)
plt.title('Спектр исходного сигнала')
plt.xlabel('Частота (Гц)')
plt.ylabel('PSD')
plt.grid(True)
plt.xlim(0, max_audio_freq*1.5)

# Спектр восстановленного сигнала
plt.subplot(2, 1, 2)
f, Pxx_filt = signal.welch(filtered_signal, fs, nperseg=1024)
plt.semilogy(f, Pxx_filt)
plt.title('Спектр восстановленного сигнала')
plt.xlabel('Частота (Гц)')
plt.ylabel('PSD')
plt.grid(True)
plt.xlim(0, max_audio_freq*1.5)

plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(plots_dir, "pdm_methods_comparison.png"), dpi=150)
plt.close()  # Закрываем текущее окно графика, чтобы избежать наложения

# Функция для улучшенного PDM кодирования с дельта-сигма модуляцией
def delta_sigma_pdm(audio_signal):
    pdm_output = np.zeros_like(audio_signal)
    error = 0
    
    for i in range(len(audio_signal)):
        # Добавляем ошибку квантования к входному сигналу
        # Это ключевая особенность дельта-сигма модуляции - формирование спектра шума
        input_plus_error = audio_signal[i] + error
        
        # Квантование
        if input_plus_error >= 0:
            pdm_output[i] = 1
        else:
            pdm_output[i] = 0
        
        # Вычисляем ошибку квантования и "запоминаем" ее для следующего отсчета
        # Это позволяет "вытолкнуть" шум квантования в высокочастотную область
        quantized = 2 * pdm_output[i] - 1  # Преобразуем 0/1 в -1/+1
        error = input_plus_error - quantized
    
    return pdm_output

# Применяем улучшенный алгоритм PDM
improved_pdm_signal = delta_sigma_pdm(audio_signal)
improved_optical_signal = transmit_optical(improved_pdm_signal, t)
improved_received_signal = photodetector_comparator(improved_optical_signal)
# Применяем тот же фильтр Баттерворта для дельта-сигма PDM сигнала
improved_filtered_signal = lfilter(b, a, improved_received_signal)

# Удаляем постоянную составляющую (DC offset)
improved_filtered_signal = improved_filtered_signal - np.mean(improved_filtered_signal)

# Нормализуем улучшенный восстановленный сигнал
if np.max(np.abs(improved_filtered_signal)) > 0:
    improved_filtered_signal = improved_filtered_signal / np.max(np.abs(improved_filtered_signal))

# Сравнение результатов базового PDM и улучшенного PDM с дельта-сигма модуляцией
plt.figure(figsize=(14, 12))

# Исходный сигнал
plt.subplot(5, 1, 1)
plt.plot(t, audio_signal)
plt.title('Исходный аудио сигнал')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.grid(True)

# Базовый PDM сигнал (фрагмент)
plt.subplot(5, 1, 2)
show_samples = min(1000, len(t))
plt.plot(t[:show_samples], pdm_signal[:show_samples], 'r-')
plt.title('Базовый PDM сигнал (фрагмент)')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.grid(True)

# Дельта-сигма PDM сигнал (фрагмент)
plt.subplot(5, 1, 3)
plt.plot(t[:show_samples], improved_pdm_signal[:show_samples], 'g-')
plt.title('Дельта-сигма PDM сигнал (фрагмент)')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.grid(True)

# Восстановленный сигнал с базовым PDM
plt.subplot(5, 1, 4)
plt.plot(t, filtered_signal)
plt.title('Восстановленный сигнал с базовым PDM')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.grid(True)

# Восстановленный сигнал с улучшенным PDM (дельта-сигма модуляция)
plt.subplot(5, 1, 5)
plt.plot(t, improved_filtered_signal)
plt.title('Восстановленный сигнал с дельта-сигма модуляцией')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.grid(True)

plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(plots_dir, "long_distance_signals.png"), dpi=150)
plt.close()  # Закрываем текущее окно графика, чтобы избежать наложения

# Добавим анализ спектра для сравнения обоих методов
plt.figure(figsize=(14, 10))

# Спектр исходного сигнала
plt.subplot(3, 1, 1)
f, Pxx_orig = signal.welch(audio_signal, fs, nperseg=1024)
plt.semilogy(f[:len(f)//10], Pxx_orig[:len(f)//10])  # Показываем только нижнюю часть спектра
plt.title('Спектр исходного сигнала')
plt.xlabel('Частота (Гц)')
plt.ylabel('PSD')
plt.grid(True)

# Спектр восстановленного сигнала с базовым PDM
plt.subplot(3, 1, 2)
f, Pxx_filt = signal.welch(filtered_signal, fs, nperseg=1024)
plt.semilogy(f[:len(f)//10], Pxx_filt[:len(f)//10])
plt.title('Спектр восстановленного сигнала с базовым PDM')
plt.xlabel('Частота (Гц)')
plt.ylabel('PSD')
plt.grid(True)

# Спектр восстановленного сигнала с дельта-сигма модуляцией
plt.subplot(3, 1, 3)
f, Pxx_improved = signal.welch(improved_filtered_signal, fs, nperseg=1024)
plt.semilogy(f[:len(f)//10], Pxx_improved[:len(f)//10])
plt.title('Спектр восстановленного сигнала с дельта-сигма модуляцией')
plt.xlabel('Частота (Гц)')
plt.ylabel('PSD')
plt.grid(True)

plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(plots_dir, "spectrum_analysis.png"), dpi=150)
plt.close()  # Закрываем текущее окно графика, чтобы избежать наложения

# Расчет SNR для обоих методов
def calculate_snr(original, reconstructed):
    # Нормализация сигналов
    original = original / np.max(np.abs(original))
    reconstructed = reconstructed / np.max(np.abs(reconstructed))
    
    # Вычисляем мощность сигнала
    signal_power = np.mean(original**2)
    
    # Вычисляем мощность шума
    noise = original - reconstructed
    noise_power = np.mean(noise**2)
    
    # Вычисляем SNR в дБ
    if noise_power > 0:
        snr = 10 * np.log10(signal_power / noise_power)
    else:
        snr = float('inf')  # Идеальная реконструкция
    
    return snr

# Рассчитываем SNR для обоих методов
snr_basic_pdm = calculate_snr(audio_signal, filtered_signal)
snr_improved_pdm = calculate_snr(audio_signal, improved_filtered_signal)

print(f"SNR для базового PDM: {snr_basic_pdm:.2f} дБ")
print(f"SNR для улучшенного PDM (дельта-сигма): {snr_improved_pdm:.2f} дБ")

# Параметры физической модели лазерной системы
WAVELENGTH = 905e-9  # Длина волны лазера (905 нм)
LASER_POWER = 15  # Увеличиваем с 15 Вт до 50 Вт
LASER_DIVERGENCE_ANGLE = 14  # Уменьшаем угол для лучшей концентрации энергии 5
TRANSMISSION_DISTANCE = 300.0  # Расстояние передачи в метрах
PULSE_FREQUENCY = fs  # Частота следования импульсов

# Создание экземпляра лазерного передатчика
laser_transmitter = Transmitter(
    wavelength=WAVELENGTH,
    power=LASER_POWER,
    beam_divergence_parallel=LASER_DIVERGENCE_ANGLE,
    beam_divergence_perpendicular=LASER_DIVERGENCE_ANGLE,
    divergence_in_degrees=True,
    pulse_duration=pulse_duration*1e9,  # переводим секунды в наносекунды
    pulse_frequency=PULSE_FREQUENCY,
)

# Создание экземпляра фотоприемника
photodiode_receiver = Receiver.from_photodiode_model("MTAPD-06-010", bias_voltage=-15) # C30902 S5973 C30737MH MTAPD-06-010

def long_distance_optical_transmission(pdm_signal, t):
    """
    Моделирует передачу PDM сигнала через оптический канал на большое расстояние
    с учетом сложного атмосферного затухания (например, туман, дождь).
    """
    bit_samples = int(pulse_duration * fs)
    bit_samples = max(1, bit_samples)
    optical_signal = np.zeros_like(pdm_signal, dtype=float)

    # Характеристики луча на расстоянии
    power_density = laser_transmitter.get_power_density(TRANSMISSION_DISTANCE)
    beam_radius_parallel, beam_radius_perpendicular = laser_transmitter.get_beam_radius(TRANSMISSION_DISTANCE)
    beam_area = laser_transmitter.get_beam_area(TRANSMISSION_DISTANCE)

    # --- модель атмосферного затухания ---
    # Коэффициенты экстинкции (примерные значения, 1/м)
    alpha_clear = 0.01 / 1000    # Чистый воздух (0.01 1/км)
    alpha_fog = 0.2 / 1000       # Легкий туман (0.2 1/км)
    alpha_heavy_fog = 0.5 / 1000 # Сильный туман (0.5 1/км)
    alpha_rain = 0.1 / 1000      # Дождь (0.1 1/км)

    # Выберите нужный коэффициент затухания
    alpha = alpha_fog  # Например, легкий туман

    # Экспоненциальное затухание по закону Бугера-Ламберта-Бера
    atmospheric_attenuation = np.exp(-alpha * TRANSMISSION_DISTANCE)

    for i in range(len(pdm_signal)):
        if pdm_signal[i] == 1:
            start_idx = i
            end_idx = min(i + bit_samples, len(optical_signal))
            actual_power_density = power_density * atmospheric_attenuation
            optical_signal[start_idx:end_idx] = actual_power_density

    return optical_signal, {
        'power_density': power_density,
        'beam_radius_parallel': beam_radius_parallel,
        'beam_radius_perpendicular': beam_radius_perpendicular,
        'beam_area': beam_area
    }

# Параметры трансимпедансного усилителя LMH34400
TIA_GAIN = 40e3  # Коэффициент усиления, 40 кОм
TIA_OUTPUT_NOISE_DENSITY = 94e-9  # В/√Гц (94 нВ/√Гц) - выходная плотность шума
TIA_BANDWIDTH = 240e6  # Полоса пропускания 240 МГц
TIA_INPUT_IMPEDANCE = 50  # Входное сопротивление, Ом

def optical_filter_transmission(wavelength):
    """
    Рассчитывает коэффициент пропускания оптического фильтра для заданной длины волны.
    Использует функцию Гаусса для моделирования характеристики пропускания.
    """
    # Гауссова функция для моделирования пропускания вблизи центральной длины волны
    sigma = OPTICAL_FILTER_BANDWIDTH / (2 * np.sqrt(2 * np.log(2)))  # Преобразование FWHM в сигму
    transmission = OPTICAL_FILTER_TRANSMISSION * np.exp(-((wavelength - OPTICAL_FILTER_CENTER)**2) / (2 * sigma**2))
    
    # Учитываем минимальное пропускание вне полосы (OD фактор)
    min_transmission = OPTICAL_FILTER_TRANSMISSION * 10**(-OPTICAL_FILTER_OD)
    transmission = max(transmission, min_transmission)
    
    return transmission

def long_distance_photodetector(optical_signal):
    """
    Моделирует прием сигнала фотодиодом с оптическим фильтром на большом расстоянии.
    """
    # Фоновая засветка для солнечного дня
    background_illumination = 0.05  # Вт/м² (солнечный день)
    
    # Коэффициенты пропускания для лазера и фоновой засветки
    laser_transmission = optical_filter_transmission(WAVELENGTH)
    
    # Для фоновой засветки учитываем средневзвешенное пропускание по спектру солнца
    # Упрощённо: в среднем внеполосное пропускание (OD4 = 0.0001) для большей части спектра
    background_transmission = 10**(-OPTICAL_FILTER_OD)
    
    # Вычисляем мощность на фотодиоде с учётом фильтра
    receiver_area = photodiode_receiver.area
    
    # Лазерный сигнал проходит через фильтр с потерями
    receiver_power = optical_signal * receiver_area * laser_transmission
    
    # Фоновая засветка сильно ослабляется фильтром
    background_power = background_illumination * receiver_area * background_transmission
    receiver_power += background_power
    
    # Преобразуем оптическую мощность в электрический ток
    sensitivity = photodiode_receiver.get_sensitivity()
    photocurrent = receiver_power * sensitivity
      
    # Добавляем темновой ток фотодиода
    photocurrent += photodiode_receiver.dark_current
    
    # Добавляем шумы фотодиода
    noise_current = photodiode_receiver.calculate_noise_current()
    shot_noise = np.random.normal(0, noise_current, size=len(photocurrent))
    bg_shot_noise = np.sqrt(2 * 1.602e-19 * sensitivity * background_power * fs)
    bg_noise = np.random.normal(0, bg_shot_noise, size=len(photocurrent))
    
    photocurrent += shot_noise + bg_noise
    
    # Трансимпедансный усилитель LMH34400
    voltage_signal = photocurrent * TIA_GAIN
    
    # Используем только эффективную полосу для аудио сигнала при расчете шума
    effective_bandwidth = min(TIA_BANDWIDTH, max_audio_freq * 2)
    tia_output_noise_rms = TIA_OUTPUT_NOISE_DENSITY * np.sqrt(effective_bandwidth)
    tia_noise = np.random.normal(0, tia_output_noise_rms, size=len(voltage_signal))
    voltage_signal += tia_noise
    
    # Используем исходный сигнал без фильтрации для компаратора
    filtered_voltage = voltage_signal
    
    # Определение порога с использованием гистограммы
    hist, bin_edges = np.histogram(filtered_voltage, bins=100)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    peaks, _ = signal.find_peaks(hist, height=np.max(hist)/10, distance=20)
    
    if len(peaks) >= 2:
        low_peak = bin_centers[peaks[0]]
        high_peak = bin_centers[peaks[-1]]
        threshold = (low_peak + high_peak) / 2
    else:
        threshold = np.median(filtered_voltage) + 0.1 * np.std(filtered_voltage)
    
    digital_output = (filtered_voltage > threshold).astype(float)
    
    return {
        'photocurrent': photocurrent,
        'voltage': voltage_signal,
        'filtered_voltage': filtered_voltage,
        'digital': digital_output,
        'threshold': threshold,
        'background_current': background_power * sensitivity,
        'tia_noise_contribution': tia_output_noise_rms,
        'effective_bandwidth': effective_bandwidth,
        'laser_filter_transmission': laser_transmission,
        'background_filter_transmission': background_transmission
    }

# PDM кодирование с использованием дельта-сигма модуляции
pdm_signal = delta_sigma_pdm(audio_signal)

# Передача через физическую модель на 300 метров
optical_signal, beam_info = long_distance_optical_transmission(pdm_signal, t)

# Прием и обработка сигнала
reception_results = long_distance_photodetector(optical_signal)
received_signal = reception_results['digital']

# Восстановление аудио с использованием RC-фильтра
cutoff_freq = max_audio_freq  # Частота среза 20 кГц
filtered_signal_rc = rc_filter(received_signal, cutoff_freq, fs)

# Восстановление аудио с использованием фильтра Баттерворта
order = 2  # Порядок фильтра
nyquist = 0.5 * fs
normal_cutoff = cutoff_freq / nyquist
b, a = butter(order, normal_cutoff, btype='low', analog=False)
filtered_signal_butter = lfilter(b, a, received_signal)

# Удаляем постоянную составляющую и нормализуем сигналы
filtered_signal_rc = filtered_signal_rc - np.mean(filtered_signal_rc)
if np.max(np.abs(filtered_signal_rc)) > 0:
    filtered_signal_rc = filtered_signal_rc / np.max(np.abs(filtered_signal_rc))

filtered_signal_butter = filtered_signal_butter - np.mean(filtered_signal_butter)
if np.max(np.abs(filtered_signal_butter)) > 0:
    filtered_signal_butter = filtered_signal_butter / np.max(np.abs(filtered_signal_butter))

# Расчет SNR для обоих фильтров
snr_rc = calculate_snr(audio_signal, filtered_signal_rc)
snr_butter = calculate_snr(audio_signal, filtered_signal_butter)

# Визуализация результатов с расстояния 300 метров
plt.figure(figsize=(15, 24))  # Увеличиваем высоту графика еще больше

# Исходный аудио сигнал
plt.subplot(9, 1, 1)
plt.plot(t, audio_signal)
plt.title('Исходный аудио сигнал')
plt.grid(True)
# Убираем подписи осей X для всех графиков кроме последнего
plt.tick_params(labelbottom=False)

# PDM сигнал (фрагмент)
plt.subplot(9, 1, 2)
show_samples = min(1000, len(t))
plt.plot(t[:show_samples], pdm_signal[:show_samples])
plt.title('PDM сигнал (дельта-сигма модуляция)')
plt.grid(True)
plt.tick_params(labelbottom=False)  # Убираем подписи осей X

# Оптический сигнал
plt.subplot(9, 1, 3)
plt.plot(t[:show_samples], optical_signal[:show_samples])
plt.title(f'Оптический сигнал на расстоянии {TRANSMISSION_DISTANCE}м')
plt.grid(True)
plt.tick_params(labelbottom=False)  # Убираем подписи осей X

# Фототок
plt.subplot(9, 1, 4)
plt.plot(t[:show_samples], reception_results['photocurrent'][:show_samples]*1e9)
plt.title(f'Фототок {photodiode_receiver.name} (нА)')
plt.grid(True)
plt.axhline(y=reception_results['background_current']*1e9, color='r', linestyle='--', label='Фоновый ток')
plt.legend()
plt.tick_params(labelbottom=False)  # Убираем подписи осей X

# Напряжение после TIA
plt.subplot(9, 1, 5)
plt.plot(t[:show_samples], reception_results['voltage'][:show_samples]*1000)
plt.title('Напряжение после TIA (мВ)')
plt.grid(True)
plt.tick_params(labelbottom=False)  # Убираем подписи осей X

# Отфильтрованное напряжение перед компаратором
plt.subplot(9, 1, 6)
plt.plot(t[:show_samples], reception_results['filtered_voltage'][:show_samples]*1000)
plt.axhline(y=reception_results['threshold']*1000, color='r', linestyle='--', label='Порог')
plt.title('Напряжение перед компаратором (мВ)')
plt.grid(True)
plt.legend()
plt.tick_params(labelbottom=False)  # Убираем подписи осей X

# Цифровой сигнал после компаратора
plt.subplot(9, 1, 7)
plt.plot(t[:show_samples], reception_results['digital'][:show_samples])
plt.title('Цифровой сигнал после компаратора')
plt.grid(True)
plt.tick_params(labelbottom=False)  # Убираем подписи осей X

# Восстановленный сигнал с RC-фильтром
plt.subplot(9, 1, 8)
plt.plot(t, filtered_signal_rc)
plt.title(f'Сигнал с RC-фильтром (SNR: {snr_rc:.2f} дБ)')
plt.grid(True)
plt.tick_params(labelbottom=False)  # Убираем подписи осей X

# Восстановленный сигнал с фильтром Баттерворта
plt.subplot(9, 1, 9)
plt.plot(t, filtered_signal_butter)
plt.title(f'Сигнал с фильтром Баттерворта (SNR: {snr_butter:.2f} дБ)')
plt.xlabel('Время (с)')  # Добавляем подпись оси X только для последнего графика
plt.grid(True)

# Используем tight_layout с дополнительными отступами и больше места между подграфиками
plt.tight_layout(pad=1.8, h_pad=2.0)
# plt.show()
plt.savefig(os.path.join(plots_dir, "long_distance_signals.png"), dpi=150)
plt.close()  # Закрываем текущее окно графика, чтобы избежать наложения

# Анализ спектра для обоих фильтров
plt.figure(figsize=(15, 14))  # Увеличиваем высоту с 10 до 14

# Спектр исходного сигнала
plt.subplot(3, 1, 1)
f, Pxx_orig = signal.welch(audio_signal, fs, nperseg=1024)
plt.semilogy(f, Pxx_orig)
plt.title('Спектр исходного сигнала')
plt.ylabel('PSD')
# Убираем метку оси X для первого графика, чтобы избежать наложения
# plt.xlabel('Частота (Гц)')  
plt.grid(True)
plt.xlim(0, max_audio_freq*1.5)

# Спектр сигнала с RC-фильтром
plt.subplot(3, 1, 2)
f, Pxx_rc = signal.welch(filtered_signal_rc, fs, nperseg=1024)
plt.semilogy(f, Pxx_rc)
plt.title('Спектр сигнала с RC-фильтром')  # Упрощаем заголовок
plt.ylabel('PSD')
# Убираем метку оси X для второго графика
# plt.xlabel('Частота (Гц)')  
plt.grid(True)
plt.xlim(0, max_audio_freq*1.5)

# Спектр сигнала с фильтром Баттерворта
plt.subplot(3, 1, 3)
f, Pxx_butter = signal.welch(filtered_signal_butter, fs, nperseg=1024)
plt.semilogy(f, Pxx_butter)
plt.title('Спектр сигнала с фильтром Баттерворта')  # Упрощаем заголовок
plt.xlabel('Частота (Гц)')  # Оставляем только на последнем графике
plt.ylabel('PSD')
plt.grid(True)
plt.xlim(0, max_audio_freq*1.5)

# Добавляем общее примечание внизу графика
plt.figtext(0.5, 0.01, f"Расстояние передачи: {TRANSMISSION_DISTANCE}м", 
            ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

plt.tight_layout(pad=1.5, h_pad=2.0)
plt.subplots_adjust(bottom=0.08)  # Дополнительное место внизу для примечания
# plt.show()
plt.savefig(os.path.join(plots_dir, "spectrum_analysis.png"), dpi=150)
plt.close()  # Закрываем текущее окно графика, чтобы избежать наложения

# Вывод информации о системе и результатах
print(f"\nПараметры системы связи на расстоянии {TRANSMISSION_DISTANCE} метров:")
print(f"Лазер: {LASER_POWER} Вт, длина волны {WAVELENGTH*1e9:.1f} нм, угол расходимости {LASER_DIVERGENCE_ANGLE}°")

# Добавляем информацию о трансимпедансном усилителе LMH34400
print(f"Трансимпедансный усилитель LMH34400:")
print(f"  - Коэффициент усиления: {TIA_GAIN/1e3:.1f} кОм")
print(f"  - Полоса пропускания: {TIA_BANDWIDTH/1e6:.0f} МГц")
print(f"  - Выходная плотность шума: {TIA_OUTPUT_NOISE_DENSITY*1e9:.1f} нВ/√Гц")
print(f"  - Эффективная полоса для аудио: {reception_results['effective_bandwidth']/1e3:.1f} кГц")
print(f"  - Шум в полосе аудио: {reception_results['tia_noise_contribution']*1e6:.2f} мкВ RMS")

print(f"Размер пятна на расстоянии {TRANSMISSION_DISTANCE}м:")
print(f"  - Радиус по горизонтали: {beam_info['beam_radius_parallel']:.2f} м")
print(f"  - Радиус по вертикали: {beam_info['beam_radius_perpendicular']:.2f} м")
print(f"  - Площадь пятна: {beam_info['beam_area']:.2f} м²")
print(f"Плотность мощности на приемнике: {beam_info['power_density']*1e6:.6f} мкВт/м²")

print("\nРезультаты восстановления сигнала:")
print(f"SNR с RC-фильтром: {snr_rc:.2f} дБ")
print(f"SNR с фильтром Баттерворта: {snr_butter:.2f} дБ")
print(f"Разница SNR: {abs(snr_butter - snr_rc):.2f} дБ в пользу {('RC-фильтра' if snr_rc > snr_butter else 'фильтра Баттерворта')}")

# Функция сохранения результатов в TXT с явным указанием кодировки UTF-8
def save_results_to_txt(filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Параметры системы связи на расстоянии {TRANSMISSION_DISTANCE} метров:\n")
        f.write(f"Лазер: {LASER_POWER} Вт, длина волны {WAVELENGTH*1e9:.1f} нм, угол расходимости {LASER_DIVERGENCE_ANGLE}°\n\n")
        
        f.write(f"Трансимпедансный усилитель LMH34400:\n")
        f.write(f"  - Коэффициент усиления: {TIA_GAIN/1e3:.1f} кОм\n")
        f.write(f"  - Полоса пропускания: {TIA_BANDWIDTH/1e6:.0f} МГц\n")
        f.write(f"  - Выходная плотность шума: {TIA_OUTPUT_NOISE_DENSITY*1e9:.1f} нВ/√Гц\n")
        f.write(f"  - Эффективная полоса для аудио: {reception_results['effective_bandwidth']/1e3:.1f} кГц\n")
        f.write(f"  - Шум в полосе аудио: {reception_results['tia_noise_contribution']*1e6:.2f} мкВ RMS\n\n")
        
        f.write(f"Размер пятна на расстоянии {TRANSMISSION_DISTANCE}м:\n")
        f.write(f"  - Радиус по горизонтали: {beam_info['beam_radius_parallel']:.2f} м\n")
        f.write(f"  - Радиус по вертикали: {beam_info['beam_radius_perpendicular']:.2f} м\n")
        f.write(f"  - Площадь пятна: {beam_info['beam_area']:.2f} м²\n")
        f.write(f"Плотность мощности на приемнике: {beam_info['power_density']*1e6:.6f} мкВт/м²\n\n")
        
        f.write(f"Результаты восстановления сигнала:\n")
        f.write(f"SNR с RC-фильтром: {snr_rc:.2f} дБ\n")
        f.write(f"SNR с фильтром Баттерворта: {snr_butter:.2f} дБ\n")
        f.write(f"Разница SNR: {abs(snr_butter - snr_rc):.2f} дБ в пользу {('RC-фильтра' if snr_rc > snr_butter else 'фильтра Баттерворта')}\n")

# Сохранение результатов в текстовый файл
results_filename = os.path.join(results_dir, f"results_dist_{int(TRANSMISSION_DISTANCE)}m.txt")
save_results_to_txt(results_filename)

print(f"\nГрафики сохранены в директорию: {plots_dir}")
print(f"Результаты сохранены в файл: {results_filename}")