import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import soundfile as sf
from scipy.signal import lfilter, butter
from IPython.display import Audio

# Параметры симуляции
fs_audio = 44100            # Частота дискретизации аудио в Гц
fs_pdm = 3000000           # Частота PDM (10 МГц) - длительность импульса PDM 100 нс
fs_optical = 333333333      # Частота оптического канала (333.33 МГц) - импульсы драйвера 3 нс
duration = 0.01             # Длительность сигнала в секундах (уменьшена для сохранения памяти)
max_audio_freq = 20000      # Максимальная частота звука в Гц (20 кГц)
pulse_duration = 3e-9       # Длительность импульса драйвера в секундах (3 нс)

# Функция для генерации тестового аудио сигнала
def generate_audio_signal(t):
    # Создаем сигнал с несколькими частотами
    f1, f2, f3 = 1000, 5000, 15000  # частоты в Гц
    signal = 0.5 * np.sin(2 * np.pi * f1 * t) + 0.3 * np.sin(2 * np.pi * f2 * t) + 0.2 * np.sin(2 * np.pi * f3 * t)
    return signal

# Функция для PDM кодирования
def pdm_encode(audio_signal):
    # Инициализируем переменные
    pdm_output = np.zeros_like(audio_signal)
    integrator = 0
    
    # Процесс PDM кодирования
    for i in range(len(audio_signal)):
        if audio_signal[i] > integrator:
            pdm_output[i] = 1
        else:
            pdm_output[i] = 0
        
        # Обновляем интегратор
        integrator = integrator + (pdm_output[i] - audio_signal[i]) * 0.1
    
    return pdm_output

# Функция для преобразования PDM импульсов в оптические импульсы драйвера (3 нс)
def laser_driver(pdm_signal, pdm_fs, optical_fs):
    # Соотношение частот дискретизации PDM и оптического сигнала
    upsampling_ratio = optical_fs // pdm_fs
    
    # Создаем высокочастотный массив для оптического сигнала
    optical_signal_length = len(pdm_signal) * upsampling_ratio
    optical_signal = np.zeros(optical_signal_length)
    
    # Количество отсчетов для 3 нс импульса при частоте optical_fs
    pulse_samples = max(1, int(pulse_duration * optical_fs))
    
    # Для каждого бита PDM сигнала
    for i in range(len(pdm_signal)):
        if pdm_signal[i] == 1:
            # Найдем начало соответствующего участка в выходном массиве
            start_idx = i * upsampling_ratio
            # Создаем короткий импульс (3 нс) в начале каждого PDM импульса
            end_idx = min(start_idx + pulse_samples, len(optical_signal))
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
    alpha = 1 / (1 + fs * tau)  # Правильный коэффициент RC-фильтра
    
    # Применяем фильтр
    filtered_signal = np.zeros_like(signal_in)
    filtered_signal[0] = signal_in[0]
    
    for i in range(1, len(signal_in)):
        filtered_signal[i] = alpha * signal_in[i] + (1 - alpha) * filtered_signal[i-1]
    
    return filtered_signal

# Генерируем временную ось для аудио
t_audio = np.linspace(0, duration, int(fs_audio * duration), endpoint=False)

# Генерируем исходный аудио сигнал
audio_signal = generate_audio_signal(t_audio)

# Нормализуем аудио сигнал
audio_signal = audio_signal / np.max(np.abs(audio_signal))

# Повышаем частоту дискретизации аудио до частоты PDM
upsample_factor_pdm = fs_pdm // fs_audio
t_pdm = np.linspace(0, duration, int(fs_pdm * duration), endpoint=False)
audio_pdm_rate = signal.resample(audio_signal, len(t_pdm))

# Нормализуем апсемплированный сигнал
audio_pdm_rate = audio_pdm_rate / np.max(np.abs(audio_pdm_rate))

# PDM кодирование
pdm_signal = pdm_encode(audio_pdm_rate)

# Драйвер лазера (преобразует PDM импульсы в оптические импульсы 3 нс)
optical_signal = laser_driver(pdm_signal, fs_pdm, fs_optical)

# Временная ось для оптического сигнала
t_optical = np.linspace(0, duration, len(optical_signal), endpoint=False)

# Фотоприемник с компаратором
received_signal = photodetector_comparator(optical_signal)

# Понижаем частоту дискретизации после фотоприемника для применения RC-фильтра
# (для экономии вычислительных ресурсов)
downsample_factor = fs_optical // fs_pdm
received_signal_downsampled = received_signal[::downsample_factor]

# RC фильтр для восстановления аудио
cutoff_freq = 20000  # Частота среза фильтра (20 кГц)
filtered_signal_rc = rc_filter(received_signal_downsampled, cutoff_freq, fs_pdm)

# Правильное восстановление биполярного сигнала:
# 1. Сначала удаляем DC-составляющую (среднее значение)
filtered_signal_rc = filtered_signal_rc - np.mean(filtered_signal_rc)

# 2. Масштабируем к полному диапазону [-1, 1]
if np.max(np.abs(filtered_signal_rc)) > 0:
    filtered_signal_rc = filtered_signal_rc / np.max(np.abs(filtered_signal_rc))

# Альтернативный вариант для восстановления биполярности:
# filtered_signal_rc = 2 * (filtered_signal_rc - 0.5)  # Преобразование из [0,1] в [-1,1]

# Построение графиков
plt.figure(figsize=(15, 15))

# 1. Исходный аудио сигнал
plt.subplot(5, 1, 1)
plt.plot(t_audio, audio_signal)
plt.title('Исходный аудио сигнал')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.grid(True)

# 2. PDM сигнал (импульсы 100 нс)
plt.subplot(5, 1, 2)
show_samples = min(1000, len(t_pdm))
plt.plot(t_pdm[:show_samples], pdm_signal[:show_samples], 'r-')
plt.title('PDM сигнал (импульсы 100 нс, первые 1000 отсчетов)')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.grid(True)

# 3. Оптический сигнал от драйвера лазера (импульсы 3 нс)
plt.subplot(5, 1, 3)
# Показываем только очень маленький участок для наглядности импульсов 3 нс
show_optical_samples = min(1000, len(t_optical))
plt.plot(t_optical[:show_optical_samples] * 1e6, optical_signal[:show_optical_samples], 'g-')
plt.title('Оптический сигнал от драйвера лазера (импульсы 3 нс)')
plt.xlabel('Время (мкс)')
plt.ylabel('Амплитуда')
plt.grid(True)

# 4. Сигнал после фотоприемника и компаратора (понижение частоты дискретизации)
plt.subplot(5, 1, 4)
plt.plot(t_pdm[:show_samples], received_signal_downsampled[:show_samples], 'b-')
plt.title('Сигнал после фотоприемника и компаратора')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.grid(True)

# 5. Восстановленный сигнал после RC-фильтра
plt.subplot(5, 1, 5)
plt.plot(t_pdm, filtered_signal_rc)
plt.title('Восстановленный сигнал после RC-фильтра')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.grid(True)

plt.tight_layout()
plt.show()

# Тестирование разных частот среза RC-фильтра
cutoff_frequencies = [5000, 10000, 20000, 50000]
plt.figure(figsize=(15, 10))

for i, cutoff in enumerate(cutoff_frequencies):
    filtered = rc_filter(received_signal_downsampled, cutoff, fs_pdm)
    filtered = filtered - np.mean(filtered)  # Удаляем DC-составляющую
    filtered = filtered / np.max(np.abs(filtered)) if np.max(np.abs(filtered)) > 0 else filtered
    
    plt.subplot(len(cutoff_frequencies), 1, i+1)
    plt.plot(t_pdm, filtered)
    plt.title(f'RC-фильтр с частотой среза {cutoff/1000} кГц')
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')
    plt.grid(True)

plt.tight_layout()
plt.show()

# Анализ спектра
plt.figure(figsize=(14, 8))

# Спектр исходного сигнала
plt.subplot(2, 1, 1)
f, Pxx_orig = signal.welch(audio_signal, fs_audio, nperseg=1024)
plt.semilogy(f, Pxx_orig)
plt.title('Спектр исходного сигнала')
plt.xlabel('Частота (Гц)')
plt.ylabel('PSD')
plt.grid(True)
plt.xlim(0, max_audio_freq*1.5)

# Спектр восстановленного сигнала
plt.subplot(2, 1, 2)
f_rc, Pxx_rc = signal.welch(filtered_signal_rc, fs_pdm, nperseg=1024)
plt.semilogy(f_rc[:len(f_rc)//10], Pxx_rc[:len(f_rc)//10])  # Показываем только нижнюю часть спектра
plt.title('Спектр восстановленного сигнала (RC-фильтр)')
plt.xlabel('Частота (Гц)')
plt.ylabel('PSD')
plt.grid(True)
plt.xlim(0, max_audio_freq*1.5)

plt.tight_layout()
plt.show()

# Расчет SNR
def calculate_snr(original, reconstructed, orig_fs, recon_fs):
    # Приводим сигналы к одной частоте дискретизации (к частоте оригинального сигнала)
    if orig_fs != recon_fs:
        resampled = signal.resample(reconstructed, len(original))
    else:
        resampled = reconstructed
    
    # Нормализация сигналов
    original = original / np.max(np.abs(original))
    resampled = resampled / np.max(np.abs(resampled))
    
    # Вычисляем мощность сигнала
    signal_power = np.mean(original**2)
    
    # Вычисляем мощность шума
    noise = original - resampled
    noise_power = np.mean(noise**2)
    
    # Вычисляем SNR в дБ
    if noise_power > 0:
        snr = 10 * np.log10(signal_power / noise_power)
    else:
        snr = float('inf')  # Идеальная реконструкция
    
    return snr

# Рассчитываем SNR для RC-фильтра
resampled_rc_signal = signal.resample(filtered_signal_rc, len(audio_signal))
snr_rc = calculate_snr(audio_signal, resampled_rc_signal, fs_audio, fs_audio)
print(f"SNR для RC-фильтра: {snr_rc:.2f} дБ")