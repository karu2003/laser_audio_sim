import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal


class PPMSimulation:
    def __init__(
        self,
        fs=44100,              # Частота дискретизации аудио, Гц
        audio_max_freq=20000,  # Максимальная частота аудио, Гц
        pulse_width=3e-9,      # Длительность импульса, с (3 нс)
        duty_ratio=0.1,        # Коэффициент заполнения, %
        min_pulse_interval=None,  # Минимальный интервал между импульсами, с (автоматический расчет)
        pico_timer_resolution=7.5e-9,  # Разрешение таймера Raspberry Pi Pico (7.5 нс)
        fixed_frame_duration=None,  # Фиксированная длительность фрейма, с (если None, рассчитывается автоматически)
        window_size=2,         # Размер окна для декодирования (в количестве фреймов)
        snr_db=30,             # Отношение сигнал/шум, дБ
    ):
        self.fs = fs
        self.audio_max_freq = audio_max_freq
        self.pulse_width = pulse_width
        self.duty_ratio = duty_ratio / 100.0  # Переводим из % в десятичную дробь
        self.window_size = window_size

        # Автоматический расчет минимального интервала между импульсами
        if min_pulse_interval is None:
            # Формула: min_pulse_interval = pulse_width * (1/duty_ratio - 1)
            self.min_pulse_interval = pulse_width * (1 / self.duty_ratio - 1)
            print(
                f"Автоматически рассчитанный минимальный интервал: {self.min_pulse_interval*1e6:.3f} мкс"
            )
        else:
            self.min_pulse_interval = min_pulse_interval

        self.pico_timer_resolution = pico_timer_resolution
        self.snr_db = snr_db

        # Расчет минимальной длительности кадра PPM с учетом структуры:
        # 1. Стартовый импульс
        # 2. Защитная зона
        # 3. Зона для информационного импульса
        # 4. Защитная зона

        # 1. Минимальная длительность кадра с учетом duty ratio и двух импульсов
        min_frame_for_duty = 2 * pulse_width / self.duty_ratio
        
        # 2. Минимальная длительность кадра с учетом минимальных интервалов
        # Импульс + защитная зона + информационный импульс + защитная зона
        min_frame_for_interval = 2 * pulse_width + 2 * self.min_pulse_interval
        
        # 3. Минимальная длительность с учетом разрядности для 10-битного аудио
        # Нужно обеспечить 1024 позиции для информационного импульса
        min_frame_for_resolution = (2 * pulse_width + 
                                    2 * self.min_pulse_interval + 
                                    1024 * pico_timer_resolution)
        
        # Определяем минимально необходимую длительность фрейма
        min_required_frame = max(min_frame_for_duty, min_frame_for_interval, min_frame_for_resolution)
        
        # Если фиксированная длительность фрейма не указана, используем минимально необходимую
        # с запасом 10% для надежности
        if fixed_frame_duration is None:
            self.frame_duration = min_required_frame * 1.1
        else:
            # Проверяем, что указанная длительность не меньше минимально необходимой
            if fixed_frame_duration < min_required_frame:
                print(f"Предупреждение: указанная фиксированная длительность фрейма ({fixed_frame_duration*1e6:.3f} мкс) " 
                      f"меньше минимально необходимой ({min_required_frame*1e6:.3f} мкс). "
                      f"Будет использовано минимальное значение.")
                self.frame_duration = min_required_frame
            else:
                self.frame_duration = fixed_frame_duration

        # Количество доступных позиций для информационного импульса
        # Учитываем два защитных интервала и два импульса
        available_space = self.frame_duration - 2 * pulse_width - 2 * self.min_pulse_interval
        self.positions_per_frame = max(2, int(np.floor(available_space / pico_timer_resolution)))
        
        # Ограничиваем позиции до 1024 для 10-битного аудио
        self.positions_per_frame = min(self.positions_per_frame, 1024)
        
        # Расчет частоты дискретизации для PPM сигнала
        self.ppm_fs = max(fs * 100, int(1 / (pulse_width / 10)))
        
        # Количество отсчетов на кадр PPM
        self.samples_per_frame = int(self.frame_duration * self.ppm_fs)
        
        # Количество отсчетов на импульс PPM
        self.samples_per_pulse = max(1, int(pulse_width * self.ppm_fs))
        
        # Минимальный интервал между импульсами в отсчетах
        self.min_interval_samples = max(1, int(self.min_pulse_interval * self.ppm_fs))
        
        # Размер окна для декодирования в отсчетах (теперь всего 2 фрейма)
        self.window_samples = self.window_size * self.samples_per_frame
        
        # Эффективный битрейт (бит/с)
        self.effective_bitrate = np.log2(self.positions_per_frame) / self.frame_duration
        
        print(f"Параметры симуляции с фиксированной длиной фрейма:")
        print(f"- Частота дискретизации аудио: {self.fs} Гц")
        print(f"- Максимальная частота аудио: {self.audio_max_freq} Гц")
        print(f"- Длительность импульса лазера: {self.pulse_width*1e9} нс")
        print(f"- Коэффициент заполнения лазера: {self.duty_ratio*100} %")
        print(f"- Минимальный интервал между импульсами: {self.min_pulse_interval*1e6} мкс")
        print(f"- Период повторения импульсов при макс. частоте: {(self.pulse_width / self.duty_ratio)*1e6} мкс")
        print(f"- Разрешение таймера Raspberry Pi Pico: {self.pico_timer_resolution*1e9} нс")
        print(f"- Фиксированная длительность фрейма: {self.frame_duration*1e6} мкс")
        print(f"- Частота фреймов: {1/self.frame_duration:.2f} Гц")
        print(f"- Количество позиций для информационного импульса: {self.positions_per_frame} (10 бит: {1024})")
        print(f"- Битовая глубина аудио: {np.log2(self.positions_per_frame):.1f} бит")
        print(f"- Эффективный битрейт: {self.effective_bitrate/1000:.2f} кбит/с")
        print(f"- Частота дискретизации PPM: {self.ppm_fs} Гц")
        print(f"- Количество отсчетов на фрейм: {self.samples_per_frame}")
        print(f"- Количество отсчетов на импульс: {self.samples_per_pulse}")
        print(f"- Мин. интервал между импульсами: {self.min_interval_samples} отсчетов")
        print(f"- Размер окна декодирования: {self.window_size} фреймов")

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
        """Преобразование аудио сигнала в PPM с фиксированной длиной фрейма"""
        # Расчет количества фреймов PPM для данного аудио
        total_duration = len(audio) / self.fs
        frames_count = max(1, int(np.floor(total_duration / self.frame_duration)))
        
        # Ресемплинг аудио для соответствия частоте фреймов PPM
        resampled_audio = signal.resample(audio, frames_count)
        print(f"audio_to_ppm: frames_count={frames_count}, resampled_audio.shape={resampled_audio.shape}")

        # Нормализация от -1...1 в 0...1
        if np.max(np.abs(resampled_audio)) > 0:
            normalized_audio = (resampled_audio - np.min(resampled_audio)) / (
                np.max(resampled_audio) - np.min(resampled_audio)
            )
        else:
            normalized_audio = np.zeros_like(resampled_audio)

        # Преобразование амплитуды в позицию импульса
        positions = np.clip(
            np.round(normalized_audio * (self.positions_per_frame - 1)),
            0,
            self.positions_per_frame - 1,
        ).astype(int)

        # Создание PPM сигнала со строго фиксированной структурой фрейма:
        # 1. Стартовый импульс 
        # 2. Защитная зона
        # 3. Информационный импульс (положение кодирует данные)
        # 4. Защитная зона
        # 5. Начало следующего фрейма
        
        ppm_signal = np.zeros(frames_count * self.samples_per_frame)

        for i, pos in enumerate(positions):
            frame_start = i * self.samples_per_frame
            
            # 1. Стартовый импульс в начале фрейма (фиксированная позиция)
            start_pulse_start = frame_start
            start_pulse_end = min(start_pulse_start + self.samples_per_pulse, len(ppm_signal))
            ppm_signal[start_pulse_start:start_pulse_end] = 1.0
            
            # 2. Защитная зона после стартового импульса
            guard_end = start_pulse_end + self.min_interval_samples
            
            # 3. Определяем диапазон, где может находиться информационный импульс
            data_zone_start = guard_end  # Начало зоны после защитной зоны
            
            # Конец зоны данных: учитываем требуемую защитную зону перед следующим фреймом
            data_zone_end = frame_start + self.samples_per_frame - self.samples_per_pulse - self.min_interval_samples
            
            # Доступное пространство для позиционирования информационного импульса
            available_samples = data_zone_end - data_zone_start
            
            # Рассчитываем позицию с учетом доступного пространства
            if self.positions_per_frame > 1 and available_samples > 0:
                pulse_position = int(pos * available_samples / (self.positions_per_frame - 1))
            else:
                pulse_position = 0
                
            # Начало информационного импульса
            data_pulse_start = data_zone_start + pulse_position
            data_pulse_end = min(data_pulse_start + self.samples_per_pulse, data_zone_end)
            
            # Проверка, чтобы импульс не выходил за границы допустимой зоны
            if data_pulse_end <= data_zone_end:
                ppm_signal[data_pulse_start:data_pulse_end] = 1.0

        print(f"audio_to_ppm: positions.shape={positions.shape}, ppm_signal.shape={ppm_signal.shape}")
        return ppm_signal, positions

    def add_channel_noise(self, ppm_signal):
        """Добавление шума канала передачи (оптический путь)"""
        # Рассчитываем мощность сигнала
        signal_power = np.mean(ppm_signal**2)

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
        frames = np.array_split(
            comparator_output, len(comparator_output) // self.samples_per_frame
        )

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
                available_space = (
                    self.samples_per_frame
                    - self.samples_per_pulse
                    - self.min_interval_samples
                )

                # Относительная позиция от минимального значения
                relative_pos = data_pulse_pos - (start_pulse_pos + min_data_start)

                # Преобразуем в позицию из диапазона 0-(positions_per_frame-1)
                if available_space > 0 and self.positions_per_frame > 1:
                    position = int(
                        relative_pos * (self.positions_per_frame - 1) / available_space
                    )
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

    def detect_ppm_pulses_windowed(self, comparator_output):
        """Обнаружение импульсов PPM в скользящем окне с учетом возможных потерь импульсов"""
        detected_positions = []
        
        # Увеличиваем перекрытие для большей надежности при малом окне
        window_overlap = int(self.samples_per_frame * 1.5)  # Перекрытие в 1.5 фрейма
        step_size = max(1, self.window_samples - window_overlap)  # Защита от нулевого или отрицательного шага
        num_windows = max(1, int((len(comparator_output) - self.window_samples) / step_size) + 1)
        
        print(f"Декодирование в {num_windows} окнах, размер окна: {self.window_size} фреймов")
        
        # Определяем параметры для скользящего окна
        window_overlap = self.samples_per_frame  # Перекрытие окон (1 фрейм)
        step_size = self.window_samples - window_overlap
        num_windows = max(1, int((len(comparator_output) - self.window_samples) / step_size) + 1)
        
        print(f"Декодирование в {num_windows} окнах, размер окна: {self.window_size} фреймов")
        
        # Обработка каждого окна
        for window_idx in range(num_windows):
            # Определяем границы текущего окна
            window_start = window_idx * step_size
            window_end = min(window_start + self.window_samples, len(comparator_output))
            
            # Если окно слишком маленькое, пропускаем
            if window_end - window_start < self.samples_per_frame:
                continue
            
            # Получаем фрагмент сигнала для текущего окна
            window_signal = comparator_output[window_start:window_end]
            
            # Находим все импульсы в окне
            all_pulses = []
            i = 0
            while i < len(window_signal):
                if window_signal[i] > 0.5:
                    # Нашли начало импульса
                    pulse_start = i
                    # Ищем конец импульса
                    while i < len(window_signal) and window_signal[i] > 0.5:
                        i += 1
                    pulse_end = i - 1
                    
                    # Центр импульса
                    pulse_center = (pulse_start + pulse_end) // 2
                    all_pulses.append(pulse_center)
                else:
                    i += 1
            
            # Если импульсов мало, пропускаем окно
            if len(all_pulses) < 1:
                continue
            
            # Проходим по всем импульсам, считая их потенциальными стартовыми
            i = 0
            window_positions = []
            
            while i < len(all_pulses):
                start_pulse_pos = all_pulses[i]
                
                # Рассчитываем границы зоны, где ожидаем информационный импульс
                min_data_pos = start_pulse_pos + self.samples_per_pulse + self.min_interval_samples
                max_data_pos = min(start_pulse_pos + self.samples_per_frame - self.samples_per_pulse - self.min_interval_samples,
                                  len(window_signal) - 1)
                
                # Проверяем, хватает ли места для информационного импульса
                if max_data_pos <= min_data_pos:
                    # Недостаточно места, пропускаем этот импульс
                    i += 1
                    continue
                
                # Ищем ближайший импульс в ожидаемой зоне данных
                data_pulse_candidates = [p for p in all_pulses if min_data_pos <= p <= max_data_pos]
                
                if data_pulse_candidates:
                    # Нашли информационный импульс
                    data_pulse_pos = data_pulse_candidates[0]
                    
                    # Вычисляем относительную позицию в фрейме
                    relative_pos = data_pulse_pos - min_data_pos
                    available_space = max_data_pos - min_data_pos
                    
                    # Преобразуем в позицию из диапазона 0-(positions_per_frame-1)
                    if available_space > 0 and self.positions_per_frame > 1:
                        position = int(relative_pos * (self.positions_per_frame - 1) / available_space)
                        position = np.clip(position, 0, self.positions_per_frame - 1)
                    else:
                        position = 0
                    
                    window_positions.append(position)
                    
                    # Ищем следующий импульс, который может быть стартовым для нового фрейма
                    # Он должен быть примерно на расстоянии длины фрейма
                    expected_next_frame_pos = start_pulse_pos + self.samples_per_frame
                    next_frame_tolerance = self.samples_per_frame * 0.1  # 10% погрешность
                    
                    next_frame_candidates = [p for p in all_pulses 
                                           if abs(p - expected_next_frame_pos) < next_frame_tolerance]
                    
                    if next_frame_candidates:
                        # Находим индекс следующего стартового импульса в all_pulses
                        next_i = all_pulses.index(next_frame_candidates[0])
                        i = next_i  # Переходим к следующему стартовому импульсу
                    else:
                        # Ищем любой импульс после максимальной позиции данных + защитная зона
                        next_start_pos = max_data_pos + self.min_interval_samples
                        next_start_candidates = [p for p in all_pulses if p > next_start_pos]
                        
                        if next_start_candidates:
                            next_i = all_pulses.index(next_start_candidates[0])
                            i = next_i
                        else:
                            # Больше нет кандидатов, выходим из цикла
                            break
                else:
                    # Информационный импульс не найден, считаем что этот бит потерян
                    # и добавляем значение по умолчанию (0)
                    window_positions.append(0)
                    
                    # Ищем следующий импульс после максимальной зоны данных
                    next_start_pos = max_data_pos + self.min_interval_samples
                    next_start_candidates = [p for p in all_pulses if p > next_start_pos]
                    
                    if next_start_candidates:
                        next_i = all_pulses.index(next_start_candidates[0])
                        i = next_i
                    else:
                        # Больше нет кандидатов, выходим из цикла
                        break
            
            # Добавляем обнаруженные в этом окне позиции к общему результату
            detected_positions.extend(window_positions)
        
        detected_positions = np.array(detected_positions)
        print(f"detect_ppm_pulses_windowed: обнаружено {len(detected_positions)} позиций")
        return detected_positions

    def detect_ppm_pulses_improved(self, comparator_output):
        """Улучшенный метод обнаружения импульсов PPM с учетом пропущенных импульсов"""
        detected_positions = []
        
        # Находим все импульсы в сигнале
        all_pulses = []
        i = 0
        while i < len(comparator_output):
            if comparator_output[i] > 0.5:
                # Нашли начало импульса
                pulse_start = i
                # Ищем конец импульса
                while i < len(comparator_output) and comparator_output[i] > 0.5:
                    i += 1
                pulse_end = i - 1
                
                # Центр импульса
                pulse_center = (pulse_start + pulse_end) // 2
                all_pulses.append(pulse_center)
            else:
                i += 1
        
        if len(all_pulses) < 1:
            print("Не обнаружено ни одного импульса")
            return np.array([])
        
        # Проходим по всем импульсам, считая их потенциальными стартовыми
        i = 0
        while i < len(all_pulses):
            start_pulse_pos = all_pulses[i]
            
            # Рассчитываем границы зоны, где ожидаем информационный импульс
            min_data_pos = start_pulse_pos + self.samples_per_pulse + self.min_interval_samples
            max_data_pos = min(start_pulse_pos + self.samples_per_frame - self.samples_per_pulse - self.min_interval_samples,
                               len(comparator_output) - 1)
            
            # Ищем ближайший импульс в ожидаемой зоне данных
            data_pulse_candidates = [p for p in all_pulses if min_data_pos <= p <= max_data_pos]
            
            if data_pulse_candidates:
                # Нашли информационный импульс
                data_pulse_pos = data_pulse_candidates[0]
                
                # Вычисляем относительную позицию в фрейме
                relative_pos = data_pulse_pos - min_data_pos
                available_space = max_data_pos - min_data_pos
                
                # Преобразуем в позицию из диапазона 0-(positions_per_frame-1)
                if available_space > 0 and self.positions_per_frame > 1:
                    position = int(relative_pos * (self.positions_per_frame - 1) / available_space)
                    position = np.clip(position, 0, self.positions_per_frame - 1)
                else:
                    position = 0
                
                detected_positions.append(position)
                
                # Ищем следующий импульс, который может быть стартовым для нового фрейма
                # Он должен быть на расстоянии не менее длины фрейма
                next_frame_candidates = [p for p in all_pulses 
                                        if p > start_pulse_pos + self.samples_per_frame * 0.9]
                
                if next_frame_candidates:
                    # Находим индекс следующего стартового импульса в all_pulses
                    next_i = all_pulses.index(next_frame_candidates[0])
                    i = next_i  # Переходим к следующему стартовому импульсу
                else:
                    # Больше нет кандидатов, выходим из цикла
                    break
            else:
                # Информационный импульс не найден, считаем что этот бит потерян
                # и добавляем значение по умолчанию (0)
                detected_positions.append(0)
                
                # Ищем следующий импульс после ожидаемой зоны данных, который может быть стартовым
                next_start_candidates = [p for p in all_pulses 
                                        if p > max_data_pos]
                
                if next_start_candidates:
                    # Находим индекс следующего возможного стартового импульса
                    next_i = all_pulses.index(next_start_candidates[0])
                    i = next_i  # Переходим к следующему возможному стартовому импульсу
                else:
                    # Больше нет кандидатов, выходим из цикла
                    break
        
        detected_positions = np.array(detected_positions)
        print(f"detect_ppm_pulses_improved: обнаружено {len(detected_positions)} позиций")
        return detected_positions

    def ppm_to_audio(self, detected_positions, original_audio_length=None):
        """Преобразование обнаруженных позиций PPM обратно в аудио сигнал"""
        # Проверка на пустой массив обнаруженных позиций
        if len(detected_positions) == 0:
            print("Предупреждение: не обнаружено ни одной позиции PPM. Возвращаю тишину.")
            if original_audio_length:
                return np.zeros(original_audio_length)
            else:
                # Если длина тоже не известна, возвращаем 1 секунду тишины
                return np.zeros(int(self.fs))
                
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

    def run_simulation(self, audio=None, duration=1.0, use_windowed_detection=True):
        """Запуск полной симуляции с опцией декодирования в окне"""
        try:
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

            # 4. Обнаружение импульсов (с использованием окна или стандартного метода)
            try:
                if use_windowed_detection:
                    detected_positions = self.detect_ppm_pulses_windowed(comparator_output)
                else:
                    detected_positions = self.detect_ppm_pulses(comparator_output)
                    
                # Проверка на пустой массив
                if len(detected_positions) == 0:
                    print("Предупреждение: функция обнаружения вернула пустой массив позиций.")
                    # Создаем массив с нулями такой же длины как positions
                    detected_positions = np.zeros_like(positions)
            except Exception as e:
                print(f"Ошибка при обнаружении импульсов: {e}")
                # Возвращаем массив нулей
                detected_positions = np.zeros_like(positions)

            # 5. Преобразование PPM обратно в аудио
            recovered_audio = self.ppm_to_audio(detected_positions, original_audio_length)

            # Рассчитываем ошибку восстановления
            if len(recovered_audio) == len(audio):
                mse = np.mean((audio - recovered_audio) ** 2)
                print(f"Среднеквадратичная ошибка восстановления: {mse:.6f}")
            else:
                print(f"Warning: recovered_audio.shape={recovered_audio.shape}, audio.shape={audio.shape}")

            return {
                "original_audio": audio,
                "ppm_signal": ppm_signal,
                "noisy_ppm": noisy_ppm,
                "comparator_output": comparator_output,
                "positions": positions,
                "detected_positions": detected_positions,
                "recovered_audio": recovered_audio,
            }
        except Exception as e:
            print(f"Ошибка в процессе симуляции: {e}")
            # Возвращаем пустой результат в случае ошибки
            return {
                "original_audio": audio if audio is not None else np.zeros(int(duration * self.fs)),
                "ppm_signal": np.array([]),
                "noisy_ppm": np.array([]),
                "comparator_output": np.array([]),
                "positions": np.array([]),
                "detected_positions": np.array([]),
                "recovered_audio": np.zeros(int(duration * self.fs)),
            }

    def plot_results(self, results, plot_segments=True):
        """Визуализация результатов симуляции PPM"""
        plt.figure(figsize=(15, 10))

        # 1. Исходный аудио сигнал
        plt.subplot(5, 1, 1)
        if (plot_segments):
            segment = min(1000, len(results["original_audio"]))
            plt.plot(results["original_audio"][:segment])
        else:
            plt.plot(results["original_audio"])
        plt.title("Исходный аудио сигнал")
        plt.grid(True)

        # 2. PPM сигнал
        plt.subplot(5, 1, 2)
        if (plot_segments):
            segment = min(self.samples_per_frame * 3, len(results["ppm_signal"]))
            plt.plot(results["ppm_signal"][:segment])
            plt.title(f"PPM сигнал (показаны первые {segment/self.ppm_fs*1e6:.1f} мкс)")
        else:
            plt.plot(results["ppm_signal"])
            plt.title("PPM сигнал")
        plt.grid(True)

        # 3. Зашумленный PPM сигнал
        plt.subplot(5, 1, 3)
        if (plot_segments):
            segment = min(self.samples_per_frame * 3, len(results["noisy_ppm"]))
            plt.plot(results["noisy_ppm"][:segment])
            plt.title(
                f"Зашумленный PPM сигнал (показаны первые {segment/self.ppm_fs*1e6:.1f} мкс)"
            )
        else:
            plt.plot(results["noisy_ppm"])
            plt.title("Зашумленный PPM сигнал")
        plt.grid(True)

        # 4. Выход компаратора
        plt.subplot(5, 1, 4)
        if (plot_segments):
            segment = min(self.samples_per_frame * 3, len(results["comparator_output"]))
            plt.plot(results["comparator_output"][:segment])
            plt.title(
                f"Выход компаратора (показаны первые {segment/self.ppm_fs*1e6:.1f} мкс)"
            )
        else:
            plt.plot(results["comparator_output"])
            plt.title("Выход компаратора")
        plt.grid(True)

        # 5. Восстановленный аудио сигнал
        plt.subplot(5, 1, 5)
        if (plot_segments):
            segment = min(1000, len(results["recovered_audio"]))
            plt.plot(results["recovered_audio"][:segment])
        else:
            plt.plot(results["recovered_audio"])
        plt.title("Восстановленный аудио сигнал")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        # Дополнительный график: спектрограммы исходного и восстановленного сигналов
        plt.figure(figsize=(15, 8))

        plt.subplot(2, 1, 1)
        plt.specgram(results["original_audio"], Fs=self.fs, NFFT=1024, noverlap=512)
        plt.title("Спектрограмма исходного аудио сигнала")
        plt.colorbar(label="Интенсивность (dB)")
        plt.ylabel("Частота (Гц)")
        plt.ylim(0, self.audio_max_freq)  # Ограничиваем по макс. частоте аудио

        plt.subplot(2, 1, 2)
        plt.specgram(results["recovered_audio"], Fs=self.fs, NFFT=1024, noverlap=512)
        plt.title("Спектрограмма восстановленного аудио сигнала")
        plt.colorbar(label="Интенсивность (dB)")
        plt.xlabel("Время (с)")
        plt.ylabel("Частота (Гц)")
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
        end_idx = min(end_idx, len(results["ppm_signal"]))

        plt.plot(
            np.arange(start_idx, end_idx) / self.ppm_fs * 1e6,
            results["ppm_signal"][start_idx:end_idx],
            "b-",
        )

        plt.title(f"Детальный вид импульсов PPM ({num_frames_to_show} кадра)")
        plt.xlabel("Время (мкс)")
        plt.ylabel("Амплитуда")
        plt.grid(True)

        # Добавляем аннотации для лазерных импульсов
        for i in range(num_frames_to_show):
            if (start_idx + (i + 1) * self.samples_per_frame) <= end_idx:
                frame_start = start_idx + i * self.samples_per_frame
                pos = results["positions"][i] if i < len(results["positions"]) else 0
                pulse_position = int(
                    pos * self.samples_per_frame / self.positions_per_frame
                )
                pulse_start = frame_start + pulse_position

                # Аннотация ширины импульса
                if (pulse_start + self.samples_per_pulse) < end_idx:
                    plt.annotate(
                        f"tw = {self.pulse_width*1e9:.1f} нс",
                        xy=(
                            (pulse_start + self.samples_per_pulse / 2)
                            / self.ppm_fs
                            * 1e6,
                            0.5,
                        ),
                        xytext=(
                            (pulse_start + self.samples_per_pulse / 2)
                            / self.ppm_fs
                            * 1e6,
                            0.7,
                        ),
                        arrowprops=dict(arrowstyle="->"),
                        horizontalalignment="center",
                    )

                # Аннотация периода кадра
                if (i == 0 and (frame_start + self.samples_per_frame) < end_idx):
                    plt.annotate(
                        f"T = {self.frame_duration*1e6:.1f} мкс (DR = {self.duty_ratio}%)",
                        xy=(
                            (frame_start + self.samples_per_frame / 2)
                            / self.ppm_fs
                            * 1e6,
                            0.1,
                        ),
                        xytext=(
                            (frame_start + self.samples_per_frame / 2)
                            / self.ppm_fs
                            * 1e6,
                            -0.2,
                        ),
                        arrowprops=dict(arrowstyle="<->", shrinkA=0, shrinkB=0),
                        horizontalalignment="center",
                    )

        plt.tight_layout()
        plt.show()

        # График соответствия исходных и обнаруженных позиций
        plt.figure(figsize=(15, 6))

        # Выбираем небольшой сегмент для визуализации
        positions_to_show = min(100, len(results["positions"]))

        plt.subplot(2, 1, 1)
        plt.plot(
            results["positions"][:positions_to_show], "o-", label="Исходные позиции"
        )
        plt.plot(
            results["detected_positions"][:positions_to_show],
            "x-",
            label="Обнаруженные позиции",
        )
        plt.legend()
        plt.title("Сравнение исходных и обнаруженных позиций PPM")
        plt.grid(True)

        plt.subplot(2, 1, 2)
        error = np.abs(
            results["positions"][:positions_to_show]
            - results["detected_positions"][:positions_to_show]
        )
        plt.plot(error, "r-")
        plt.title("Ошибка обнаружения позиций")
        plt.grid(True)

        plt.tight_layout()
        plt.show()


# Запуск симуляции с учетом новых параметров
if __name__ == "__main__":
    # Создаем экземпляр симулятора с параметрами лазера
    simulator = PPMSimulation()

    # Генерируем тестовый аудио сигнал (сумма нескольких частот)
    test_audio = simulator.generate_test_audio(
        duration=0.1,  # Короткая длительность для быстрой симуляции
        frequencies=[440, 1000, 5000, 15000],  # Тестовые частоты до 20 кГц
    )

    # Запускаем симуляцию
    results = simulator.run_simulation(test_audio)

    # Визуализируем результаты
    simulator.plot_results(results, plot_segments=True)
