import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import soundfile as sf
from scipy.signal import lfilter, butter
from IPython.display import Audio
from photodiode_presets import PHOTODIODE_PRESETS
from LiFi_sim import Transmitter, Receiver

# Параметры симуляции
fs = 1000000  # Частота дискретизации в Гц (1 МГц)
duration = 0.05  # Длительность сигнала в секундах
pulse_duration = 3e-9  # Длительность импульса в секундах (3 нс)
max_audio_freq = 20000  # Максимальная частота звука в Гц (20 кГц)
oversampling_ratio = fs // (2 * max_audio_freq)  # Коэффициент передискретизации

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

# Используем более совершенный фильтр Баттерворта для сравнения с RC
# RC фильтр - это фильтр первого порядка
filtered_signal_rc = rc_filter(received_signal, cutoff_freq, fs)

# Также можно использовать фильтр Баттерворта для лучшего качества
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
plt.show()

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
plt.show()

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
plt.show()

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
plt.show()

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
LASER_POWER = 15  # Мощность лазера в Ваттах
LASER_DIVERGENCE_ANGLE = 14  # Угол расходимости в градусах
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
photodiode_receiver = Receiver.from_photodiode_model("S5973")

def long_distance_optical_transmission(pdm_signal, t):
    """
    Моделирует передачу PDM сигнала через оптический канал на большое расстояние.
    """
    # Размер одного бита в наших отсчетах
    bit_samples = int(pulse_duration * fs)
    bit_samples = max(1, bit_samples)
    
    # Создаем массив для оптического сигнала
    optical_signal = np.zeros_like(pdm_signal, dtype=float)
    
    # Рассчитываем характеристики луча на расстоянии 300 метров
    power_density = laser_transmitter.get_power_density(TRANSMISSION_DISTANCE)
    beam_radius_parallel, beam_radius_perpendicular = laser_transmitter.get_beam_radius(TRANSMISSION_DISTANCE)
    beam_area = laser_transmitter.get_beam_area(TRANSMISSION_DISTANCE)
    
    # Для каждого бита в PDM сигнале
    for i in range(len(pdm_signal)):
        if pdm_signal[i] == 1:
            # Создаем короткий импульс с учетом мощности лазера и расстояния
            start_idx = i
            end_idx = min(i + bit_samples, len(optical_signal))
            
            # Учитываем затухание на больших расстояниях
            atmospheric_attenuation = np.exp(-0.05 * TRANSMISSION_DISTANCE / 1000)
            actual_power_density = power_density * atmospheric_attenuation
            
            optical_signal[start_idx:end_idx] = actual_power_density
    
    return optical_signal, {
        'power_density': power_density,
        'beam_radius_parallel': beam_radius_parallel,
        'beam_radius_perpendicular': beam_radius_perpendicular,
        'beam_area': beam_area
    }

def long_distance_photodetector(optical_signal):
    """
    Моделирует прием сигнала фотодиодом на большом расстоянии с учетом шумов.
    """
    # Фоновая засветка
    background_illumination = 1e-6  # Вт/м²
    
    # Вычисляем мощность на фотодиоде
    receiver_area = photodiode_receiver.area
    receiver_power = optical_signal * receiver_area
    
    # Добавляем фоновую засветку
    background_power = background_illumination * receiver_area
    receiver_power += background_power
    
    # Преобразуем оптическую мощность в электрический ток
    sensitivity = photodiode_receiver.get_sensitivity()
    photocurrent = receiver_power * sensitivity
    
    # Добавляем темновой ток фотодиода
    photocurrent += photodiode_receiver.dark_current
    
    # Добавляем шумы
    noise_current = photodiode_receiver.calculate_noise_current()
    shot_noise = np.random.normal(0, noise_current, size=len(photocurrent))
    bg_shot_noise = np.sqrt(2 * 1.602e-19 * sensitivity * background_power * fs)
    bg_noise = np.random.normal(0, bg_shot_noise, size=len(photocurrent))
    
    photocurrent += shot_noise + bg_noise
    
    # Трансимпедансный усилитель
    tia_gain = 40e3  # В/А
    voltage_signal = photocurrent * tia_gain
    
    # Добавляем шум электроники TIA LMH34400
    tia_noise_density = 2.5e-15  # A/√Hz 
    tia_noise = np.random.normal(0, tia_noise_density * np.sqrt(fs/2) * tia_gain, size=len(voltage_signal))
    voltage_signal += tia_noise
    
    # Компаратор для восстановления цифрового сигнала
    avg_voltage = np.mean(voltage_signal)
    threshold = max(avg_voltage * 1.5, background_power * sensitivity * tia_gain * 2)
    digital_output = (voltage_signal > threshold).astype(float)
    
    return {
        'photocurrent': photocurrent,
        'voltage': voltage_signal,
        'digital': digital_output,
        'threshold': threshold,
        'background_current': background_power * sensitivity
    }

# Обновленная основная часть кода
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
plt.figure(figsize=(15, 20))

# Исходный аудио сигнал
plt.subplot(8, 1, 1)
plt.plot(t, audio_signal)
plt.title('Исходный аудио сигнал')
plt.grid(True)

# PDM сигнал (фрагмент)
plt.subplot(8, 1, 2)
show_samples = min(1000, len(t))
plt.plot(t[:show_samples], pdm_signal[:show_samples])
plt.title('PDM сигнал (дельта-сигма модуляция)')
plt.grid(True)

# Оптический сигнал
plt.subplot(8, 1, 3)
plt.plot(t[:show_samples], optical_signal[:show_samples])
plt.title(f'Оптический сигнал на расстоянии {TRANSMISSION_DISTANCE}м (мкВт/м²)')
plt.grid(True)

# Фототок
plt.subplot(8, 1, 4)
plt.plot(t[:show_samples], reception_results['photocurrent'][:show_samples]*1e9)
plt.title('Фототок S5973 (нА)')
plt.grid(True)
plt.axhline(y=reception_results['background_current']*1e9, color='r', linestyle='--', label='Фоновый ток')
plt.legend()

# Напряжение после TIA
plt.subplot(8, 1, 5)
plt.plot(t[:show_samples], reception_results['voltage'][:show_samples]*1000)
plt.title('Напряжение после TIA (мВ)')
plt.grid(True)
plt.axhline(y=reception_results['threshold']*1000, color='r', linestyle='--', label='Порог компаратора')
plt.legend()

# Цифровой сигнал после компаратора
plt.subplot(8, 1, 6)
plt.plot(t[:show_samples], received_signal[:show_samples])
plt.title('Цифровой сигнал после компаратора')
plt.grid(True)

# Восстановленный сигнал с RC-фильтром
plt.subplot(8, 1, 7)
plt.plot(t, filtered_signal_rc)
plt.title(f'Восстановленный сигнал с RC-фильтром (SNR: {snr_rc:.2f} дБ)')
plt.grid(True)

# Восстановленный сигнал с фильтром Баттерворта
plt.subplot(8, 1, 8)
plt.plot(t, filtered_signal_butter)
plt.title(f'Восстановленный сигнал с фильтром Баттерворта (SNR: {snr_butter:.2f} дБ)')
plt.grid(True)

plt.tight_layout()
plt.show()

# Анализ спектра для обоих фильтров
plt.figure(figsize=(15, 10))

# Спектр исходного сигнала
plt.subplot(3, 1, 1)
f, Pxx_orig = signal.welch(audio_signal, fs, nperseg=1024)
plt.semilogy(f, Pxx_orig)
plt.title('Спектр исходного сигнала')
plt.xlabel('Частота (Гц)')
plt.ylabel('PSD')
plt.grid(True)
plt.xlim(0, max_audio_freq*1.5)

# Спектр сигнала с RC-фильтром
plt.subplot(3, 1, 2)
f, Pxx_rc = signal.welch(filtered_signal_rc, fs, nperseg=1024)
plt.semilogy(f, Pxx_rc)
plt.title(f'Спектр сигнала с RC-фильтром (расстояние {TRANSMISSION_DISTANCE}м)')
plt.xlabel('Частота (Гц)')
plt.ylabel('PSD')
plt.grid(True)
plt.xlim(0, max_audio_freq*1.5)

# Спектр сигнала с фильтром Баттерворта
plt.subplot(3, 1, 3)
f, Pxx_butter = signal.welch(filtered_signal_butter, fs, nperseg=1024)
plt.semilogy(f, Pxx_butter)
plt.title(f'Спектр сигнала с фильтром Баттерворта (расстояние {TRANSMISSION_DISTANCE}м)')
plt.xlabel('Частота (Гц)')
plt.ylabel('PSD')
plt.grid(True)
plt.xlim(0, max_audio_freq*1.5)

plt.tight_layout()
plt.show()

# Вывод информации о системе и результатах
print(f"\nПараметры системы связи на расстоянии {TRANSMISSION_DISTANCE} метров:")
print(f"Лазер: {LASER_POWER} Вт, длина волны {WAVELENGTH*1e9:.1f} нм, угол расходимости {LASER_DIVERGENCE_ANGLE}°")
print(f"Размер пятна на расстоянии {TRANSMISSION_DISTANCE}м:")
print(f"  - Радиус по горизонтали: {beam_info['beam_radius_parallel']:.2f} м")
print(f"  - Радиус по вертикали: {beam_info['beam_radius_perpendicular']:.2f} м")
print(f"  - Площадь пятна: {beam_info['beam_area']:.2f} м²")
print(f"Плотность мощности на приемнике: {beam_info['power_density']*1e6:.6f} мкВт/м²")

print("\nРезультаты восстановления сигнала:")
print(f"SNR с RC-фильтром: {snr_rc:.2f} дБ")
print(f"SNR с фильтром Баттерворта: {snr_butter:.2f} дБ")
print(f"Разница SNR: {abs(snr_butter - snr_rc):.2f} дБ в пользу {('RC-фильтра' if snr_rc > snr_butter else 'фильтра Баттерворта')}")