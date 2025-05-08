import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

class PPMSimulation:
    def __init__(self, 
                 fs=44100,             # Частота дискретизации аудио, Гц
                 audio_max_freq=20000, # Максимальная частота аудио, Гц
                 pulse_width=3e-9,     # Длительность импульса, с (3 нс)
                 duty_ratio=0.1,       # Коэффициент заполнения, %
                 min_pulse_interval=1e-7, # Минимальный интервал между импульсами, с (100 нс)
                 pico_timer_resolution=7.5e-9, # Разрешение таймера Raspberry Pi Pico (7.5 нс)
                 snr_db=30):           # Отношение сигнал/шум, дБ
        
        self.fs = fs
        self.audio_max_freq = audio_max_freq
        self.pulse_width = pulse_width
        self.duty_ratio = duty_ratio
        self.min_pulse_interval = min_pulse_interval
        self.pico_timer_resolution = pico_timer_resolution
        self.snr_db = snr_db
        
        # Расчет длительности кадра PPM с учетом защитного интервала
        # между импульсами для предотвращения перегрева лазера
        # Мы используем двухимпульсную структуру: стартовый и информационный импульсы
        
        # 1. Минимальная длительность кадра с учетом duty ratio
        min_frame_for_duty = pulse_width * 100 / duty_ratio
        
        # 2. Минимальная длительность кадра с учетом минимального интервала
        # Два импульса + минимальный интервал между ними
        min_frame_for_interval = 2 * pulse_width + min_pulse_interval
        
        # 3. Минимальная длительность с учетом разрядности для 10-битного аудио
        # Для 10-битного сигнала нужно 1024 позиции
        min_frame_for_resolution = 1024 * pico_timer_resolution
        
        # Выбираем максимальное из всех ограничений
        self.frame_duration = max(min_frame_for_duty, min_frame_for_interval, min_frame_for_resolution)
        
        # Количество доступных позиций для второго импульса (с учетом разрешения Pico)
        self.positions_per_frame = max(2, int(np.floor((self.frame_duration - 2*pulse_width) / pico_timer_resolution)))
        
        # Ограничиваем позиции до 1024 для 10-битного аудио
        self.positions_per_frame = min(self.positions_per_frame, 1024)
        
        # Расчет частоты дискретизации для PPM сигнала
        self.ppm_fs = max(fs * 100, int(1 / (pulse_width / 10)))
        
        # Количество отсчетов на кадр PPM
        self.samples_per_frame = int(self.frame_duration * self.ppm_fs)
        
        # Количество отсчетов на импульс PPM
        self.samples_per_pulse = max(1, int(pulse_width * self.ppm_fs))
        
        # Минимальный интервал между импульсами в отсчетах
        self.min_interval_samples = max(1, int(min_pulse_interval * self.ppm_fs))
        
        print(f"Параметры симуляции:")
        print(f"- Частота дискретизации аудио: {self.fs} Гц")
        print(f"- Максимальная частота аудио: {self.audio_max_freq} Гц")
        print(f"- Длительность импульса лазера: {self.pulse_width*1e9} нс")
        print(f"- Коэффициент заполнения лазера: {self.duty_ratio} %")
        print(f"- Минимальный интервал между импульсами: {self.min_pulse_interval*1e6} мкс")
        print(f"- Разрешение таймера Raspberry Pi Pico: {self.pico_timer_resolution*1e9} нс")
        print(f"- Длительность кадра PPM: {self.frame_duration*1e6} мкс")
        print(f"- Частота кадров: {1/self.frame_duration:.2f} Гц")
        print(f"- Количество позиций в кадре: {self.positions_per_frame} (10 бит: {1024})")
        print(f"- Битовая глубина аудио: {np.log2(self.positions_per_frame):.1f} бит")
        print(f"- Частота дискретизации PPM: {self.ppm_fs} Гц")
        print(f"- Количество отсчетов на кадр: {self.samples_per_frame}")
        print(f"- Количество отсчетов на импульс: {self.samples_per_pulse}")
        print(f"- Мин. интервал между импульсами: {self.min_interval_samples} отсчетов")
    
    def generate_test_audio(self, duration=1.0, frequencies=[440, 1000, 5000, 15000]):
        """Генерация тестового аудио сигнала"""
        t = np.linspace(0, duration, int(duration * self.fs), endpoint=False)
        audio = np.zeros_like(t)
        
        # Суммирование нескольких синусоид разных частот
        for freq in frequencies:
            audio += np.sin(2 * np.pi * freq * t)
        
        # Нормализация
        audio = audio / np.max(np.abs(audio)) * 0.9
        
        return audio
    
    def audio_to_ppm(self, audio):
        """Преобразование аудио сигнала в PPM с двумя импульсами на кадр"""
        # Расчет количества кадров PPM для данного аудио
        total_duration = len(audio) / self.fs
        frames_count = max(1, int(np.round(total_duration / self.frame_duration)))
        if frames_count < 2:
            print(f"Warning: frames_count={frames_count}, возможно слишком мало кадров для восстановления сигнала.")
        # Ресемплинг аудио для соответствия частоте кадров PPM
        resampled_audio = signal.resample(audio, frames_count)
        print(f"audio_to_ppm: frames_count={frames_count}, resampled_audio.shape={resampled_audio.shape}")

        # Нормализация от -1...1 в 0...1
        if np.max(np.abs(resampled_audio)) > 0:
            normalized_audio = (resampled_audio - np.min(resampled_audio)) / (np.max(resampled_audio) - np.min(resampled_audio))
        else:
            normalized_audio = np.zeros_like(resampled_audio)

        # Преобразование амплитуды в позицию импульса
        positions = np.clip(np.round(normalized_audio * (self.positions_per_frame - 1)),
                            0, self.positions_per_frame - 1).astype(int)

        # Создание PPM сигнала с двумя импульсами на кадр
        ppm_signal = np.zeros(frames_count * self.samples_per_frame)
        
        for i, pos in enumerate(positions):
            frame_start = i * self.samples_per_frame
            
            # Стартовый импульс в начале кадра
            start_pulse_start = frame_start
            start_pulse_end = min(start_pulse_start + self.samples_per_pulse, len(ppm_signal))
            ppm_signal[start_pulse_start:start_pulse_end] = 1.0
            
            # Информационный импульс с учетом минимального интервала
            # Позиция начинается после минимального интервала от конца стартового импульса
            min_data_start = start_pulse_end + self.min_interval_samples
            
            # Доступное пространство для позиционирования информационного импульса
            available_space = self.samples_per_frame - self.samples_per_pulse - self.min_interval_samples
            
            # Рассчитываем позицию с учетом доступного пространства
            if self.positions_per_frame > 1:
                pulse_position = int(pos * available_space / (self.positions_per_frame - 1))
            else:
                pulse_position = 0
                
            # Начало информационного импульса
            data_pulse_start = min_data_start + pulse_position
            data_pulse_end = min(data_pulse_start + self.samples_per_pulse, frame_start + self.samples_per_frame)
            
            # Проверка, чтобы импульс не выходил за границы кадра
            if data_pulse_end <= frame_start + self.samples_per_frame:
                ppm_signal[data_pulse_start:data_pulse_end] = 1.0

        print(f"audio_to_ppm: positions.shape={positions.shape}, ppm_signal.shape={ppm_signal.shape}")
        return ppm_signal, positions
    
    def add_channel_noise(self, ppm_signal):
        """Добавление шума канала передачи (оптический путь)"""
        # Рассчитываем мощность сигнала
        signal_power = np.mean(ppm_signal ** 2)
        
        # Рассчитываем мощность шума на основе SNR
        noise_power = signal_power / (10 ** (self.snr_db / 10))
        
        # Добавляем белый гауссовский шум
        noise = np.random.normal(0, np.sqrt(noise_power), len(ppm_signal))
        noisy_signal = ppm_signal + noise
        
        return noisy_signal
    
    def apply_comparator(self, noisy_signal, threshold=0.5):
        """Применение компаратора к зашумленному сигналу"""
        return (noisy_signal > threshold).astype(float)
    
    def detect_ppm_pulses(self, comparator_output):
        """Обнаружение импульсов в выходном сигнале компаратора с учетом двухимпульсной структуры"""
        # Разделение на кадры
        frames = np.array_split(comparator_output, len(comparator_output) // self.samples_per_frame)
        
        detected_positions = []
        
        for frame in frames:
            # Если длина кадра слишком мала, пропускаем
            if len(frame) < self.samples_per_pulse * 2 + self.min_interval_samples:
                detected_positions.append(0)
                continue
                
            # Поиск двух импульсов в кадре
            pulse_indices = []
            i = 0
            while i < len(frame):
                if frame[i] > 0.5:
                    # Нашли начало импульса
                    pulse_start = i
                    # Ищем конец импульса
                    while i < len(frame) and frame[i] > 0.5:
                        i += 1
                    pulse_end = i - 1
                    
                    # Центр импульса
                    pulse_center = (pulse_start + pulse_end) // 2
                    pulse_indices.append(pulse_center)
                else:
                    i += 1
            
            # Если обнаружено два или более импульсов
            if len(pulse_indices) >= 2:
                # Первый импульс считаем стартовым
                start_pulse_pos = pulse_indices[0]
                # Второй импульс - информационный
                data_pulse_pos = pulse_indices[1]
                
                # Вычисляем относительную позицию информационного импульса
                min_data_start = self.samples_per_pulse + self.min_interval_samples
                available_space = self.samples_per_frame - self.samples_per_pulse - self.min_interval_samples
                
                # Относительная позиция от минимального значения
                relative_pos = data_pulse_pos - (start_pulse_pos + min_data_start)
                
                # Преобразуем в позицию из диапазона 0-(positions_per_frame-1)
                if available_space > 0 and self.positions_per_frame > 1:
                    position = int(relative_pos * (self.positions_per_frame - 1) / available_space)
                    position = np.clip(position, 0, self.positions_per_frame - 1)
                else:
                    position = 0
                
                detected_positions.append(position)
            else:
                # Если не найдено достаточно импульсов, предполагаем позицию 0
                detected_positions.append(0)
        
        detected_positions = np.array(detected_positions)
        print(f"detect_ppm_pulses: detected_positions.shape={detected_positions.shape}")
        return detected_positions
    
    def ppm_to_audio(self, detected_positions, original_audio_length=None):
        """Преобразование обнаруженных позиций PPM обратно в аудио сигнал"""
        # Преобразование позиций в нормализованные значения амплитуды (0...1)
        if self.positions_per_frame > 1:
            normalized_audio = detected_positions / (self.positions_per_frame - 1)
        else:
            normalized_audio = np.zeros_like(detected_positions)
        
        # Преобразование обратно в диапазон -1...1
        recovered_audio = normalized_audio * 2 - 1
        
        # Ресемплинг до исходной частоты дискретизации аудио
        if original_audio_length:
            recovered_audio = signal.resample(recovered_audio, original_audio_length)
        else:
            # Предположим длительность по количеству кадров
            duration = len(detected_positions) * self.frame_duration
            recovered_audio = signal.resample(recovered_audio, int(duration * self.fs))
        
        print(f"ppm_to_audio: recovered_audio.shape={recovered_audio.shape}")
        return recovered_audio
    
    def run_simulation(self, audio=None, duration=1.0):
        """Запуск полной симуляции"""
        if audio is None:
            # Создаем тестовый аудио сигнал, если не передан
            audio = self.generate_test_audio(duration)
        
        # Сохраняем длину оригинального аудио для последующего восстановления
        original_audio_length = len(audio)
        print(f"run_simulation: original_audio_length={original_audio_length}")
        
        # 1. Преобразование аудио в PPM
        ppm_signal, positions = self.audio_to_ppm(audio)
        
        # 2. Пропускание через канал передачи с шумом
        noisy_ppm = self.add_channel_noise(ppm_signal)
        
        # 3. Применение компаратора (фотоприемник)
        comparator_output = self.apply_comparator(noisy_ppm)
        
        # 4. Обнаружение импульсов
        detected_positions = self.detect_ppm_pulses(comparator_output)
        
        # 5. Преобразование PPM обратно в аудио
        recovered_audio = self.ppm_to_audio(detected_positions, original_audio_length)
        
        # Рассчитываем ошибку восстановления
        if len(recovered_audio) == len(audio):
            mse = np.mean((audio - recovered_audio) ** 2)
            print(f"Среднеквадратичная ошибка восстановления: {mse:.6f}")
        else:
            print(f"Warning: recovered_audio.shape={recovered_audio.shape}, audio.shape={audio.shape}")
        
        return {
            'original_audio': audio,
            'ppm_signal': ppm_signal,
            'noisy_ppm': noisy_ppm,
            'comparator_output': comparator_output,
            'positions': positions,
            'detected_positions': detected_positions,
            'recovered_audio': recovered_audio
        }
    
    def plot_results(self, results, plot_segments=True):
        """Визуализация результатов симуляции PPM"""
        plt.figure(figsize=(15, 10))
        
        # 1. Исходный аудио сигнал
        plt.subplot(5, 1, 1)
        if plot_segments:
            segment = min(1000, len(results['original_audio']))
            plt.plot(results['original_audio'][:segment])
        else:
            plt.plot(results['original_audio'])
        plt.title('Исходный аудио сигнал')
        plt.grid(True)
        
        # 2. PPM сигнал
        plt.subplot(5, 1, 2)
        if plot_segments:
            segment = min(self.samples_per_frame * 3, len(results['ppm_signal']))
            plt.plot(results['ppm_signal'][:segment])
            plt.title(f'PPM сигнал (показаны первые {segment/self.ppm_fs*1e6:.1f} мкс)')
        else:
            plt.plot(results['ppm_signal'])
            plt.title('PPM сигнал')
        plt.grid(True)
        
        # 3. Зашумленный PPM сигнал
        plt.subplot(5, 1, 3)
        if plot_segments:
            segment = min(self.samples_per_frame * 3, len(results['noisy_ppm']))
            plt.plot(results['noisy_ppm'][:segment])
            plt.title(f'Зашумленный PPM сигнал (показаны первые {segment/self.ppm_fs*1e6:.1f} мкс)')
        else:
            plt.plot(results['noisy_ppm'])
            plt.title('Зашумленный PPM сигнал')
        plt.grid(True)
        
        # 4. Выход компаратора
        plt.subplot(5, 1, 4)
        if plot_segments:
            segment = min(self.samples_per_frame * 3, len(results['comparator_output']))
            plt.plot(results['comparator_output'][:segment])
            plt.title(f'Выход компаратора (показаны первые {segment/self.ppm_fs*1e6:.1f} мкс)')
        else:
            plt.plot(results['comparator_output'])
            plt.title('Выход компаратора')
        plt.grid(True)
        
        # 5. Восстановленный аудио сигнал
        plt.subplot(5, 1, 5)
        if plot_segments:
            segment = min(1000, len(results['recovered_audio']))
            plt.plot(results['recovered_audio'][:segment])
        else:
            plt.plot(results['recovered_audio'])
        plt.title('Восстановленный аудио сигнал')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Дополнительный график: спектрограммы исходного и восстановленного сигналов
        plt.figure(figsize=(15, 8))
        
        plt.subplot(2, 1, 1)
        plt.specgram(results['original_audio'], Fs=self.fs, NFFT=1024, noverlap=512)
        plt.title('Спектрограмма исходного аудио сигнала')
        plt.colorbar(label='Интенсивность (dB)')
        plt.ylabel('Частота (Гц)')
        plt.ylim(0, self.audio_max_freq)  # Ограничиваем по макс. частоте аудио
        
        plt.subplot(2, 1, 2)
        plt.specgram(results['recovered_audio'], Fs=self.fs, NFFT=1024, noverlap=512)
        plt.title('Спектрограмма восстановленного аудио сигнала')
        plt.colorbar(label='Интенсивность (dB)')
        plt.xlabel('Время (с)')
        plt.ylabel('Частота (Гц)')
        plt.ylim(0, self.audio_max_freq)  # Ограничиваем по макс. частоте аудио
        
        plt.tight_layout()
        plt.show()
        
        # Отображаем детали импульса и периода для лазера
        plt.figure(figsize=(15, 5))
        
        # Показываем несколько кадров
        start_idx = 0
        num_frames_to_show = 3
        end_idx = start_idx + num_frames_to_show * self.samples_per_frame
        
        # Ограничиваем, если выходим за пределы
        end_idx = min(end_idx, len(results['ppm_signal']))
        
        plt.plot(np.arange(start_idx, end_idx) / self.ppm_fs * 1e6, 
                results['ppm_signal'][start_idx:end_idx], 'b-')
        
        plt.title(f'Детальный вид импульсов PPM ({num_frames_to_show} кадра)')
        plt.xlabel('Время (мкс)')
        plt.ylabel('Амплитуда')
        plt.grid(True)
        
        # Добавляем аннотации для лазерных импульсов
        for i in range(num_frames_to_show):
            if (start_idx + (i+1)*self.samples_per_frame) <= end_idx:
                frame_start = start_idx + i * self.samples_per_frame
                pos = results['positions'][i] if i < len(results['positions']) else 0
                pulse_position = int(pos * self.samples_per_frame / self.positions_per_frame)
                pulse_start = frame_start + pulse_position
                
                # Аннотация ширины импульса
                if pulse_start + self.samples_per_pulse < end_idx:
                    plt.annotate(
                        f'tw = {self.pulse_width*1e9:.1f} нс', 
                        xy=((pulse_start + self.samples_per_pulse/2) / self.ppm_fs * 1e6, 0.5),
                        xytext=((pulse_start + self.samples_per_pulse/2) / self.ppm_fs * 1e6, 0.7),
                        arrowprops=dict(arrowstyle="->"),
                        horizontalalignment='center'
                    )
                
                # Аннотация периода кадра
                if i == 0 and (frame_start + self.samples_per_frame) < end_idx:
                    plt.annotate(
                        f'T = {self.frame_duration*1e6:.1f} мкс (DR = {self.duty_ratio}%)', 
                        xy=((frame_start + self.samples_per_frame/2) / self.ppm_fs * 1e6, 0.1),
                        xytext=((frame_start + self.samples_per_frame/2) / self.ppm_fs * 1e6, -0.2),
                        arrowprops=dict(arrowstyle="<->", shrinkA=0, shrinkB=0),
                        horizontalalignment='center'
                    )
        
        plt.tight_layout()
        plt.show()
        
        # График соответствия исходных и обнаруженных позиций
        plt.figure(figsize=(15, 6))
        
        # Выбираем небольшой сегмент для визуализации
        positions_to_show = min(100, len(results['positions']))
        
        plt.subplot(2, 1, 1)
        plt.plot(results['positions'][:positions_to_show], 'o-', label='Исходные позиции')
        plt.plot(results['detected_positions'][:positions_to_show], 'x-', label='Обнаруженные позиции')
        plt.legend()
        plt.title('Сравнение исходных и обнаруженных позиций PPM')
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        error = np.abs(results['positions'][:positions_to_show] - results['detected_positions'][:positions_to_show])
        plt.plot(error, 'r-')
        plt.title('Ошибка обнаружения позиций')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

# Запуск симуляции с учетом новых параметров
if __name__ == "__main__":
    # Создаем экземпляр симулятора с параметрами лазера
    simulator = PPMSimulation(
        fs=44100,              # Частота дискретизации аудио, Гц
        audio_max_freq=20000,  # Максимальная частота аудио, Гц
        pulse_width=3e-9,      # Длительность импульса, с (3 нс)
        duty_ratio=0.1,        # Коэффициент заполнения, %
        min_pulse_interval=1e-7, # Минимальный интервал между импульсами, с (100 нс)
        pico_timer_resolution=7.5e-9, # Разрешение таймера Raspberry Pi Pico (7.5 нс)
        snr_db=25              # Отношение сигнал/шум, дБ
    )
    
    # Генерируем тестовый аудио сигнал (сумма нескольких частот)
    test_audio = simulator.generate_test_audio(
        duration=0.1,  # Короткая длительность для быстрой симуляции
        frequencies=[440, 1000, 5000, 15000]  # Тестовые частоты до 20 кГц
    )
    
    # Запускаем симуляцию
    results = simulator.run_simulation(test_audio)
    
    # Визуализируем результаты
    simulator.plot_results(results, plot_segments=True)