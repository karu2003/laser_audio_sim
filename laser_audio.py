import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

# Параметры системы
audio_freq = 20000  # Максимальная частота аудиосигнала (20 кГц)
pdm_freq = 10e6      # Частота PDM модуляции (10 МГц)
pdm_pulse_width = 100e-9  # Длительность импульса PDM (100 нс)
laser_pulse_width = 3e-9   # Длительность импульса лазера (3 нс)
sim_duration = 50e-6  # Длительность симуляции (50 мкс)
rc_time_constant = 1/(2*np.pi*audio_freq)  # Постоянная времени RC фильтра

# Генерация тестового аудиосигнала (синус 5 кГц)
t_audio = np.linspace(0, sim_duration, int(sim_duration * audio_freq * 10))
audio_signal = np.sin(2 * np.pi * 5000 * t_audio)  # 5 кГц синус

# PDM модуляция
def pdm_modulate(audio, pdm_freq, duration):
    t_pdm = np.linspace(0, duration, int(duration * pdm_freq))
    pdm_signal = np.zeros_like(t_pdm)
    integrator = 0
    
    # Ресемплируем аудиосигнал на PDM сетку
    audio_resampled = np.interp(t_pdm, t_audio, audio)
    
    for i in range(len(t_pdm)):
        integrator += audio_resampled[i]
        if integrator > 0:
            pdm_signal[i] = 1
            integrator -= 1
        else:
            pdm_signal[i] = 0
    return t_pdm, pdm_signal

# Моделирование драйвера лазера (преобразование в короткие импульсы)
def laser_driver(pdm_signal, pulse_width, pdm_freq):
    samples_per_pulse = int(pulse_width * pdm_freq)
    laser_signal = np.zeros_like(pdm_signal)
    for i in range(len(pdm_signal)):
        if pdm_signal[i] > 0.5:
            laser_signal[i:i+samples_per_pulse] = 1
    return laser_signal

# Моделирование фотоприемника и компаратора
def photo_receiver(laser_signal, threshold=0.5):
    # Добавляем шум и затухание
    noisy_signal = laser_signal + np.random.normal(0, 0.1, len(laser_signal))
    # Компаратор
    comparator_output = (noisy_signal > threshold).astype(float)
    return comparator_output

# RC фильтр низких частот
def rc_filter(input_signal, time_constant, sample_rate):
    alpha = 1 / (time_constant * sample_rate + 1)
    filtered = np.zeros_like(input_signal)
    filtered[0] = input_signal[0]
    for i in range(1, len(input_signal)):
        filtered[i] = alpha * input_signal[i] + (1 - alpha) * filtered[i-1]
    return filtered

# Выполняем симуляцию
t_pdm, pdm_signal = pdm_modulate(audio_signal, pdm_freq, sim_duration)
laser_signal = laser_driver(pdm_signal, laser_pulse_width, pdm_freq)
rx_signal = photo_receiver(laser_signal)
demodulated = rc_filter(rx_signal, rc_time_constant, pdm_freq)

# Ресемплируем демодулированный сигнал на аудиосетку
demodulated_audio = np.interp(t_audio, t_pdm, demodulated)

# Построение графиков
plt.figure(figsize=(12, 10))

# Исходный аудиосигнал
plt.subplot(4, 1, 1)
plt.plot(t_audio[:1000], audio_signal[:1000])
plt.title('Исходный аудиосигнал (5 кГц синус)')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.grid(True)

# PDM сигнал
plt.subplot(4, 1, 2)
plt.plot(t_pdm[:500], pdm_signal[:500])
plt.title('PDM модулированный сигнал')
plt.xlabel('Время (с)')
plt.ylabel('Уровень')
plt.grid(True)

# Сигнал после лазера и фотоприемника
plt.subplot(4, 1, 3)
plt.plot(t_pdm[:500], laser_signal[:500], label='Лазерный импульс (3 нс)')
plt.plot(t_pdm[:500], rx_signal[:500], label='После фотоприемника')
plt.title('Сигнал после лазера и фотоприемника')
plt.xlabel('Время (с)')
plt.ylabel('Уровень')
plt.legend()
plt.grid(True)

# Демодулированный сигнал
plt.subplot(4, 1, 4)
plt.plot(t_audio[:1000], demodulated_audio[:1000], label='Демодулированный')
plt.plot(t_audio[:1000], audio_signal[:1000], label='Исходный', alpha=0.5)
plt.title('Сравнение исходного и демодулированного сигналов')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()