import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from matplotlib.ticker import ScalarFormatter
from photodiode_presets import PHOTODIODE_PRESETS

# Константы
c = constants.c  # скорость света, м/с
h = constants.h  # постоянная Планка, Дж·с

# Параметры системы
# Расстояние между передатчиком и приемником
distances = np.linspace(0, 300, 1000)  # метры


# Параметры передатчика
class Transmitter:
    def __init__(
        self,
        wavelength,
        power,
        beam_divergence=None,
        pulse_duration=None,
        divergence_in_degrees=False,
        emitter_width=35e-6,
        emitter_height=10e-6,
        beam_divergence_parallel=0,
        beam_divergence_perpendicular=0,
        pulse_frequency=None,  
    ):
        self.wavelength = wavelength  # м
        self.power = power  # Вт

        # Углы расходимости по двум осям (по умолчанию в градусах)
        if divergence_in_degrees:
            self.beam_divergence_parallel = np.deg2rad(beam_divergence_parallel)
            self.beam_divergence_perpendicular = np.deg2rad(
                beam_divergence_perpendicular
            )
        else:
            self.beam_divergence_parallel = beam_divergence_parallel
            self.beam_divergence_perpendicular = beam_divergence_perpendicular

        # Для обратной совместимости (если используется старый параметр)
        if beam_divergence is not None:
            if divergence_in_degrees:
                self.beam_divergence = beam_divergence * np.pi / 180
            else:
                self.beam_divergence = beam_divergence
        else:
            self.beam_divergence = None

        # Исправлено: корректный расчет среднего угла для вывода
        if self.beam_divergence is not None:
            # Если задан скалярный угол, используем его
            self.beam_divergence_deg = self.beam_divergence * 180 / np.pi
        elif self.beam_divergence_parallel and self.beam_divergence_perpendicular:
            # Если заданы оба угла, берём среднее
            self.beam_divergence_deg = (
                (self.beam_divergence_parallel + self.beam_divergence_perpendicular)
                * 180
                / (2 * np.pi)
            )
        else:
            self.beam_divergence_deg = 0

        self.pulse_duration = pulse_duration  # наносекунды
        self.pulse_frequency = pulse_frequency  # Гц

        # Размеры излучателя
        self.emitter_width = emitter_width  # м
        self.emitter_height = emitter_height  # м
        self.emitter_area = emitter_width * emitter_height  # м²

        # Расчет энергии фотона
        self.photon_energy = h * c / wavelength  # Дж

        # Расчет количества фотонов в импульсе
        if power is not None and pulse_duration is not None:
            self.photons_per_pulse = (
                power * pulse_duration * 1e-9
            ) / self.photon_energy
        else:
            self.photons_per_pulse = None
    
    def set_power(self, power):
        """
        Изменение мощности передатчика с пересчётом зависимых параметров
        
        Args:
            power: Новое значение мощности в Ваттах
            
        Returns:
            self: Для поддержки цепочки вызовов
        """
        self.power = power  # Вт
        
        # Пересчитываем количество фотонов в импульсе, если задана длительность импульса
        if self.power is not None and self.pulse_duration is not None:
            self.photons_per_pulse = (
                self.power * self.pulse_duration * 1e-9
            ) / self.photon_energy
        
        return self

    def get_beam_radius(self, distance):
        # Радиусы эллипса на заданном расстоянии (по двум осям)
        # Формула: R = sqrt(R0^2 + (distance * tan(theta/2))^2)
        if distance == 0:
            r_parallel = self.emitter_width / 2
            r_perpendicular = self.emitter_height / 2
        else:
            r0_parallel = self.emitter_width / 2
            r0_perpendicular = self.emitter_height / 2

            # Используем либо индивидуальные значения расходимости для каждой оси,
            # либо общее значение beam_divergence, если оно задано
            if self.beam_divergence is not None:
                r_parallel = np.sqrt(
                    r0_parallel**2 + (distance * np.tan(self.beam_divergence / 2)) ** 2
                )
                r_perpendicular = np.sqrt(
                    r0_perpendicular**2
                    + (distance * np.tan(self.beam_divergence / 2)) ** 2
                )
            else:
                r_parallel = np.sqrt(
                    r0_parallel**2
                    + (distance * np.tan(self.beam_divergence_parallel / 2)) ** 2
                )
                r_perpendicular = np.sqrt(
                    r0_perpendicular**2
                    + (distance * np.tan(self.beam_divergence_perpendicular / 2)) ** 2
                )
        return r_parallel, r_perpendicular

    def get_beam_area(self, distance):
        # Площадь эллипса на заданном расстоянии
        r_parallel, r_perpendicular = self.get_beam_radius(distance)
        return np.pi * r_parallel * r_perpendicular

    def get_power_density(self, distance):
        area = self.get_beam_area(distance)
        return self.power / area

    def get_photon_flux(self, distance):
        power_density = self.get_power_density(distance)
        photon_flux = power_density / self.photon_energy
        return photon_flux

    def get_average_power(self):
        # Если заданы длительность импульса и частота следования, считаем среднюю мощность по duty cycle
        if self.pulse_duration is not None and self.pulse_frequency is not None:
            duty_cycle = (self.pulse_duration * 1e-9) * self.pulse_frequency  # pulse_duration в секундах
            duty_cycle = min(duty_cycle, 1.0)
            return self.power * duty_cycle
        return self.power


# Параметры приемника
class Receiver:
    def __init__(
        self,
        wavelength,
        area,
        sensitivity,
        quantum_efficiency,
        dark_current,
        noise_density,  # NEP в W/Hz^0.5 - эквивалентная шумовая мощность
        bias_voltage,
        bandwidth,
    ):
        self.wavelength = wavelength  # м
        self.area = area  # м²
        self.base_sensitivity = sensitivity  # A/W, базовая чувствительность
        self.quantum_efficiency = quantum_efficiency  # безразмерная 0-1
        self.dark_current = dark_current # A
        self.noise_density = noise_density  # W/Hz^0.5 - эквивалентная шумовая мощность
        self.bias_voltage = bias_voltage  # V
        self.bandwidth = bandwidth
        self.type = None
        self.name = None
    
    @classmethod
    def calculate_nep(cls, Id, F, R):
        """
        Расчет NEP (Noise Equivalent Power):
        NEP = sqrt(2 * q * Id * F) / R
        q = 1.6e-19      # заряд электрона (Кл)
        Id = 65e-9       # темновой ток (А)
        F = 0.7          # избыточный шумовой фактор
        R = 0.9          # чувствительность (A/W)
        """
        q = 1.6e-19      # заряд электрона (Кл)
        
        nep = (2 * q * Id * F) ** 0.5 / R
        return nep
        
    @classmethod
    def from_photodiode_model(cls, model_name, wavelength=None, bias_voltage=None, bandwidth=None, **kwargs):
        """
        Создаёт экземпляр Receiver на основе параметров известного фотодиода.
        
        Args:
            model_name: Название модели фотодиода из словаря PHOTODIODE_PRESETS
            wavelength: Рабочая длина волны (если не указана, используется пиковая для фотодиода)
            bias_voltage: Напряжение смещения (если не указано, используется максимальное из спецификации)
            bandwidth: Полоса пропускания (если не указана, берется из PHOTODIODE_PRESETS или 100 МГц)
            **kwargs: Дополнительные параметры для переопределения значений из PHOTODИODE_PRESETS
            
        Returns:
            Экземпляр класса Receiver с параметрами выбранного фотодиода
        """
        if model_name not in PHOTODIODE_PRESETS:
            raise ValueError(f"Неизвестная модель фотодиода: {model_name}")
        
        params = PHOTODIODE_PRESETS[model_name].copy()
        
        # Если длина волны не указана, используем пиковую
        if wavelength is None:
            wavelength = params["peak_wavelength"]
        
        # Если напряжение смещения не указано, используем максимальное
        if bias_voltage is None:
            bias_voltage = params["max_bias_voltage"]
        
        # Если полоса пропускания не указана, берем из параметров фотодиода или используем значение по умолчанию
        if bandwidth is None:
            if "bandwidth" in params:
                bandwidth = params["bandwidth"]
            else:
                bandwidth = 100e6  # Значение по умолчанию: 100 МГц
            
        # Определяем noise_density в соответствии с типом шума
        if params["noise"] == "NEP":
            noise_density = params["NEP"]  # W/Hz^0.5
        elif params["noise"] == "ENF":
            # Расчет NEP из темнового тока, избыточного шумового фактора и чувствительности
            noise_density = cls.calculate_nep(params["dark_current"], params["ENF"], params["sensitivity"])
        else:
            # Если тип шума не указан, используем значение по умолчанию
            noise_density = 1e-12  # W/Hz^0.5, типичное значение
        
        # Применяем переопределения из kwargs
        for key, value in kwargs.items():
            if key in params:
                params[key] = value
        
        receiver = cls(
            wavelength=wavelength,
            area=params["area"],
            sensitivity=params["sensitivity"],
            quantum_efficiency=params["quantum_efficiency"],
            dark_current=params["dark_current"],
            noise_density=noise_density,
            bias_voltage=bias_voltage,
            bandwidth=bandwidth
        )
        
        # Устанавливаем имя модели фотодиода
        receiver.name = model_name
        
        # Устанавливаем тип диода, если он указан в параметрах
        if "type" in params:
            receiver.type = params["type"]
        
        return receiver
    
    @classmethod
    def with_custom_area(cls, model_name, area, **kwargs):
        """
        Создаёт экземпляр Receiver на основе параметров известного фотодиода,
        но с измененной площадью.
        
        Args:
            model_name: Название модели фотодиода
            area: Кастомная площадь в м²
            **kwargs: Дополнительные параметры для from_photodiode_model
            
        Returns:
            Экземпляр класса Receiver с параметрами выбранного фотодиода и кастомной площадью
        """
        receiver = cls.from_photodiode_model(model_name, **kwargs)
        receiver.area = area
        return receiver

    def get_sensitivity(self):
        # Модель зависимости чувствительности от VBias (пример: логарифмический рост с насыщением)
        # Можно скорректировать под вашу физику
        vbias = abs(self.bias_voltage)
        max_gain = 2.5
        saturation_voltage = 150
        if vbias <= 30:
            gain = 1.0 + (vbias - 15) / 15 * 0.5
        else:
            gain = 1.5 + (max_gain - 1.5) * (
                1 - np.exp(-(vbias - 30) / saturation_voltage)
            )
        return self.base_sensitivity * min(gain, max_gain)

    def calculate_signal_current(self, incident_power):
        # Используем чувствительность, зависящую от VBias
        return incident_power * self.get_sensitivity()

    def calculate_noise_current(self):
        # Расчет шумового тока
        # Включает тепловой шум, дробовой шум и шум темнового тока
        shot_noise = np.sqrt(2 * constants.e * self.dark_current * self.bandwidth)
        
        # Преобразование NEP (W/Hz^0.5) в шумовую плотность тока (A/Hz^0.5)
        # используя чувствительность (A/W)
        # in = NEP / R
        current_noise_density = self.noise_density / self.get_sensitivity()
        thermal_noise = current_noise_density * np.sqrt(self.bandwidth)
        
        return np.sqrt(shot_noise**2 + thermal_noise**2)

    def calculate_snr(self, signal_current, noise_current):
        # Расчет отношения сигнал/шум
        if noise_current == 0:
            return float("inf")
        return 20 * np.log10(signal_current / noise_current)

    def calculate_ber(self, snr):
        # Расчет вероятности битовой ошибки
        # Используем простую модель для двоичной фазовой манипуляции
        snr_linear = 10 ** (snr / 10)
        # Ограничиваем минимальное значение BER для предотвращения нулей
        ber = 0.5 * np.exp(-snr_linear / 2)
        # Установка минимального порога для BER
        return max(ber, 1e-300)


# Моделирование с разными параметрами
def simulate_lifi_system(transmitter, receiver, distances):
    results = {}

    power_densities = []
    signal_currents = []
    noise_currents = []
    snrs = []
    bers = []

    for distance in distances:
        # Плотность мощности на приемнике
        power = transmitter.get_average_power()
        power_density = power / transmitter.get_beam_area(distance)
        power_densities.append(power_density)

        # Мощность, попадающая на приемник
        incident_power = power_density * receiver.area

        # Расчет тока сигнала
        signal_current = receiver.calculate_signal_current(incident_power)
        signal_currents.append(signal_current)

        # Расчет шумового тока
        noise_current = receiver.calculate_noise_current()
        noise_currents.append(noise_current)

        # Расчет SNR и BER
        snr = receiver.calculate_snr(signal_current, noise_current)
        snrs.append(snr)

        ber = receiver.calculate_ber(snr)
        bers.append(ber)

    results["power_densities"] = power_densities
    results["signal_currents"] = signal_currents
    results["noise_currents"] = noise_currents
    results["snrs"] = snrs
    results["bers"] = bers

    return results


# Вспомогательная функция для оценки дальности связи при заданных параметрах
def estimate_max_distance(transmitter, receiver, min_snr=10, verbose=True):
    """
    Оценка максимальной дальности связи при заданном минимальном SNR

    Args:
        transmitter: Объект передатчика
        receiver: Объект приемника
        min_snr: Минимальное требуемое SNR в дБ
        verbose: Если True, выводить предупреждения и информацию

    Returns:
        Максимальное расстояние в метрах
    """
    # Увеличиваем диапазон расстояний до 10000 метров
    distances = np.logspace(0, 4, 1000)  # от 1 до 10000 метров
    results = simulate_lifi_system(transmitter, receiver, distances)

    # Находим индекс, где SNR падает ниже минимального значения
    snr_array = np.array(results["snrs"])
    indices = np.where(snr_array < min_snr)[0]

    if len(indices) == 0:
        # Если SNR всегда выше минимального, возвращаем максимальное расстояние из диапазона
        if verbose:
            # Более информативное сообщение с параметрами передатчика
            power_mw = transmitter.power * 1000  # мВт

            # Корректный вывод угла
            div_deg = transmitter.beam_divergence_deg

            # SNR на максимальном расстоянии
            max_snr = snr_array[-1]

            if max_snr < min_snr:
                print(
                    f"Предупреждение: Для передатчика {power_mw:.1f} мВт, {div_deg:.2f}°: "
                    f"SNR = {max_snr:.1f} дБ на расстоянии {distances[-1]:.1f} м "
                    f"(мин. требуемый SNR = {min_snr} дБ)"
                )
            else:
                print(
                    f"Информация: Для передатчика {power_mw:.1f} мВт, {div_deg:.2f}°: "
                    f"SNR = {max_snr:.1f} дБ на расстоянии {distances[-1]:.1f} м "
                    f"(мин. требуемый SNR = {min_snr} дБ)"
                )
        return distances[-1]
    else:
        index = indices[0]
        if index == 0:
            return 0  # Если SNR всегда ниже минимального
        else:
            # Интерполяция для более точного определения дистанции
            d1 = distances[index - 1]
            d2 = distances[index]
            snr1 = snr_array[index - 1]
            snr2 = snr_array[index]

            # Линейная интерполяция для нахождения точки, где SNR = min_snr
            if snr1 != snr2:  # Избегаем деления на ноль
                ratio = (min_snr - snr2) / (snr1 - snr2)
                max_distance = d2 + ratio * (d1 - d2)
            else:
                max_distance = d1

            return max_distance


# Добавим функцию для анализа производительности лазера с углом 20 градусов
def analyze_high_divergence_laser(distances, model_name="BPW34", divergence = 20):
    """
    Анализ производительности лазера с высоким углом расходимости (20 градусов)

    Args:
        distances: Массив расстояний для анализа
    """
    # Создание передатчика с углом 20 градусов
    powers = [0.1, 0.5, 1.0, 2.0, 15.0, 75.0]  # Вт
    min_snr = 10  # дБ
    max_power = 75.0  # Вт, максимальная рассматриваемая мощность

    base_receiver = Receiver.from_photodiode_model(model_name)
    
    laser_20deg = Transmitter(850e-9, 1.0, divergence , 10, divergence_in_degrees=True)
    laser_1deg  = Transmitter(850e-9, 0.1, 1, 10, divergence_in_degrees=True)

    plt.figure(figsize=(12, 8))

    # Сравнение разных мощностей для лазера с углом 20 градусов
    for power in powers:
        laser_20deg.set_power(power)  # Устанавливаем мощность передатчика
        results = simulate_lifi_system(laser_20deg, base_receiver, distances)
        label = f"{power*1000:.0f} мВт, {laser_20deg.beam_divergence_deg:.1f}°"
        plt.plot(distances, results["snrs"], label=label)

    results = simulate_lifi_system(laser_1deg, base_receiver, distances)
    plt.plot(
        distances,
        results["snrs"],
        label=f"100 мВт, 1.0° (для сравнения)",
        linestyle="--",
    )

    plt.title(f"Производительность лазера с углом расходимости {divergence}°")
    plt.xlabel("Расстояние (м)")
    plt.ylabel("SNR (дБ)")
    plt.grid(True)
    plt.legend()
    plt.axhline(
        y=10, color="r", linestyle="--", label="Минимальный рекомендуемый SNR ({min_snr} дБ)"
    )
    plt.savefig("lifi_high_divergence_laser.png")
    plt.close()  # Закрываем фигуру после сохранения

    # Расчет максимальной дальности для разных мощностей
    
    print(f"Максимальная дальность для лазера с углом {divergence} градусов и приемником {base_receiver.name}:")
    for power in powers:
        laser_20deg.set_power(power) 
        max_dist = estimate_max_distance(laser_20deg, base_receiver, min_snr)
        print(f"  При мощности {power*1000:.0f} мВт: {max_dist:.1f} м")

    # Анализ влияния площади приемника
    plt.figure(figsize=(12, 8))
    receiver_areas = [1e-4, 5e-4, 1e-3, 2e-3]  # м² (1, 5, 10, 20 см²)
    
    photodiode_models = list(PHOTODIODE_PRESETS.keys())    
    # make areas list from photodiode_models area - список кортежей (имя_модели, площадь)
    areas = [(model, PHOTODIODE_PRESETS[model]["area"]) for model in photodiode_models]
    # print("Areas:", [(name, f"{area*1e6:.3f} мм²") for name, area in areas])
    
    power_1W = 1.0  # Вт

    laser_20deg.set_power(power_1W) 

    # for area in receiver_areas:
    for model_name, area in areas:    
        rx = Receiver.from_photodiode_model(model_name, area=area)  # Используем напрямую модель    
        results = simulate_lifi_system(laser_20deg, rx, distances)
        label = f"Площадь приемника {model_name}: {area*1e6:.3f} mm²"
        plt.plot(distances, results["snrs"], label=label)

    plt.title(f"Влияние площади приемника на SNR (лазер {power_1W} Вт, {divergence}°)")
    plt.xlabel("Расстояние (м)")
    plt.ylabel("SNR (дБ)")
    plt.grid(True)
    plt.legend()
    plt.axhline(
        y=10, color="r", linestyle="--", label=f"Минимальный рекомендуемый SNR ({min_snr} дБ)"
    )
    plt.savefig("lifi_receiver_area_impact.png")
    plt.close()  # Закрываем фигуру после сохранения

    # Рекомендации для достижения дальности 300 м с лазером 20°
    power_needed = None

    for power in np.linspace(0.1, max_power, 500):
        laser_20deg.set_power(power)  # Устанавливаем мощность передатчика
        max_dist = estimate_max_distance(laser_20deg, base_receiver, min_snr)
        if max_dist >= 300:
            power_needed = power
            break

    if power_needed is not None:
        print(
            f"\nДля достижения дальности 300 м с лазером {divergence}° и приемником {base_receiver.name} требуется мощность: {power_needed*1000:.0f} мВт \n"
        )
    else:
        print(
            f"\nДля достижения дальности 300 м с лазером {divergence}° и приемником {base_receiver.name} требуется мощность более {max_power*1000:.0f} мВт \n"
        )
 
    print("Различные мощности и фото диоды:")

    for power in powers:        
        for model_name, area in areas:
            laser_20deg.set_power(power)  # Устанавливаем мощность передатчика
            rx = Receiver.from_photodiode_model(model_name)  # Используем напрямую модель
            max_dist = estimate_max_distance(laser_20deg, rx, min_snr)
            if max_dist >= 300 and rx.bandwidth >= 100e6:
                print(
                    f"  Мощность {power*1000:.0f} мВт, приемник {model_name}, Bandwidth {rx.bandwidth/1e6:.0f} МГц ({area*1e6:.3f} мм²): {max_dist:.1f} м"
                )

def plot_results(distances, results, title, label):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title)

    # Плотность мощности
    axs[0, 0].plot(distances, results["power_densities"], label=label)
    axs[0, 0].set_title("Плотность мощности")
    axs[0, 0].set_xlabel("Расстояние (м)")
    axs[0, 0].set_ylabel("Плотность мощности (Вт/м²)")
    axs[0, 0].set_yscale("log")
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # Ток сигнала
    axs[0, 1].plot(
        distances, [i * 1e6 for i in results["signal_currents"]], label=label
    )
    axs[0, 1].set_title("Ток сигнала")
    axs[0, 1].set_xlabel("Расстояние (м)")
    axs[0, 1].set_ylabel("Ток (мкА)")
    axs[0, 1].set_yscale("log")
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # SNR
    axs[1, 0].plot(distances, results["snrs"], label=label)
    axs[1, 0].set_title("Отношение сигнал/шум (SNR)")
    axs[1, 0].set_xlabel("Расстояние (м)")
    axs[1, 0].set_ylabel("SNR (дБ)")
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    # BER
    # Заменяем нулевые значения на минимальное значение float
    bers_plot = [max(ber, 1e-300) for ber in results["bers"]]
    axs[1, 1].plot(distances, bers_plot, label=label)
    axs[1, 1].set_title("Вероятность битовой ошибки (BER)")
    axs[1, 1].set_xlabel("Расстояние (м)")
    axs[1, 1].set_ylabel("BER")
    axs[1, 1].set_yscale("log")
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    return fig


# Пример использования: сравнение разных лазеров
def compare_transmitters(receivers, transmitters, distances):
    for receiver in receivers:
        plt.figure(figsize=(10, 6))
        for i, transmitter in enumerate(transmitters):
            results = simulate_lifi_system(transmitter, receiver, distances)
            label = f"{transmitter.power*1000:.0f} мВт, {transmitter.beam_divergence*1000:.1f} мрад"
            plt.plot(distances, results["snrs"], label=label)

        plt.title(
            f"Сравнение передатчиков (приемник: Vbias = {receiver.bias_voltage}В)"
        )
        plt.xlabel("Расстояние (м)")
        plt.ylabel("SNR (дБ)")
        plt.grid(True)
        plt.legend()
        plt.show()


# Пример использования: сравнение разных приемников
def compare_receivers(transmitter, receivers, distances):
    plt.figure(figsize=(10, 6))
    for i, receiver in enumerate(receivers):
        results = simulate_lifi_system(transmitter, receiver, distances)
        label = (
            f"Vbias = {receiver.bias_voltage}В, QE = {receiver.quantum_efficiency:.2f}"
        )
        plt.plot(distances, results["snrs"], label=label)

    plt.title(f"Сравнение приемников (передатчик: {transmitter.power*1000:.0f} мВт)")
    plt.xlabel("Расстояние (м)")
    plt.ylabel("SNR (дБ)")
    plt.grid(True)
    plt.legend()
    plt.show()


# Пример использования: Оценка параметров системы для достижения требуемого BER
def find_transmitter_power_for_target_ber(
    receiver, wavelength, beam_divergence, pulse_duration, distances, target_ber=1e-6
):
    # Начальные параметры
    power_min = 0.001  # 1 мВт
    power_max = 5.0  # 5 Вт

    def check_ber(power):
        transmitter = Transmitter(wavelength, power, beam_divergence, pulse_duration)
        results = simulate_lifi_system(transmitter, receiver, distances)
        max_ber = max(results["bers"])
        return max_ber <= target_ber

    # Бинарный поиск для нахождения минимальной мощности
    while power_max - power_min > 0.001:
        power_mid = (power_min + power_max) / 2
        if check_ber(power_mid):
            power_max = power_mid
        else:
            power_min = power_mid

    return power_max


# Примеры параметров системы
def main():
    # Параметры передатчика
    wavelength_850nm = 850e-9  # м (ИК-диод)
    wavelength_1550nm = 1550e-9  # м (ИК-лазер, телеком)

    # Разные мощности передатчика для анализа
    laser_powers = [0.01, 0.1, 0.5, 1.0, 15.0]  # Вт

    # Расходимость луча (в радианах)
    beam_divergences = [
        0.001,
        0.005,
        0.01,
        np.radians(20),
    ]  # радианы (0.06, 0.3, 0.6 градусов)

    # Длительность импульса
    pulse_durations = [3, 10, 15]  # наносекунды

    # Создаем набор передатчиков и приемников для сравнения
    transmitters = []
    for power in laser_powers:
        for divergence in beam_divergences:
            transmitters.append(Transmitter(wavelength_850nm, power, divergence, 10))

    receivers = []
    receivers.append(Receiver.from_photodiode_model("S5973"))

    # Базовый сценарий
    base_transmitter = Transmitter(wavelength_850nm, 0.5, 0.005, 10)
    base_receiver = Receiver.from_photodiode_model("S5973")

    # Симуляция базового сценария
    base_results = simulate_lifi_system(base_transmitter, base_receiver, distances)

    # Построение графиков
    plot_results(
        distances, base_results, "Базовая модель LiFi для дрона", "Базовый сценарий"
    )
    plt.savefig("lifi_base_scenario.png")
    plt.close()  # Закрываем фигуру после сохранения

    # Сравнение различных передатчиков
    plt.figure(figsize=(10, 6))
    for power in [0.1, 0.5, 1.0]:
        for divergence in [0.001, 0.005]:
            tx = Transmitter(wavelength_850nm, power, divergence, 10)
            results = simulate_lifi_system(tx, base_receiver, distances)
            label = f"{power*1000:.0f} мВт, {tx.beam_divergence_deg:.1f}°"
            plt.plot(distances, results["snrs"], label=label)

    plt.title("Влияние мощности и расходимости луча на SNR")
    plt.xlabel("Расстояние (м)")
    plt.ylabel("SNR (дБ)")
    plt.grid(True)
    plt.legend()
    plt.savefig("lifi_transmitter_comparison.png")
    plt.close()  # Закрываем фигуру после сохранения

    # Сравнение различных приемников
    plt.figure(figsize=(10, 6))
    for bias in [5, 10, 15]:
        for qe in [0.6, 0.8]:
            rx = Receiver.from_photodiode_model("S5973", 
                bias_voltage=bias,
                quantum_efficiency=qe,
            )    
            results = simulate_lifi_system(base_transmitter, rx, distances)
            label = f"Vbias = {bias}В, QE = {qe:.2f}"
            plt.plot(distances, results["snrs"], label=label)

    plt.title("Влияние напряжения смещения и квантовой эффективности на SNR")
    plt.xlabel("Расстояние (м)")
    plt.ylabel("SNR (дБ)")
    plt.grid(True)
    plt.legend()
    plt.savefig("lifi_receiver_comparison.png")
    plt.close()  # Закрываем фигуру после сохранения

    # Анализ импульсов разной длительности
    plt.figure(figsize=(10, 6))
    for duration in [3, 10, 15]:
        tx = Transmitter(wavelength_850nm, 0.5, 0.005, duration)
        results = simulate_lifi_system(tx, base_receiver, distances)
        label = f"Длительность импульса = {duration} нс"
        plt.plot(distances, results["snrs"], label=label)

    plt.title("Влияние длительности импульса на SNR")
    plt.xlabel("Расстояние (м)")
    plt.ylabel("SNR (дБ)")
    plt.grid(True)
    plt.legend()
    plt.savefig("lifi_pulse_duration_comparison.png")
    plt.close()  # Закрываем фигуру после сохранения

    # Определение минимальной мощности для достижения целевого BER
    target_ber = 1e-6
    min_power = find_transmitter_power_for_target_ber(
        base_receiver, wavelength_850nm, 0.005, 10, distances, target_ber
    )
    print(
        f"Минимальная мощность для достижения BER {target_ber}: {min_power*1000:.1f} мВт"
    )

    # Сравнение длин волн
    plt.figure(figsize=(10, 6))
    tx_850 = Transmitter(wavelength_850nm, 0.5, 0.005, 10)
    tx_1550 = Transmitter(wavelength_1550nm, 0.5, 0.005, 10)
    
    rx_850 = Receiver.from_photodiode_model("BPW34")
    rx_1550 = Receiver.from_photodiode_model("G8931-04")

    results_850 = simulate_lifi_system(tx_850, rx_850, distances)
    results_1550 = simulate_lifi_system(tx_1550, rx_1550, distances)

    plt.plot(distances, results_850["snrs"], label="850 нм")
    plt.plot(distances, results_1550["snrs"], label="1550 нм")

    plt.title("Сравнение длин волн")
    plt.xlabel("Расстояние (м)")
    plt.ylabel("SNR (дБ)")
    plt.grid(True)
    plt.legend()
    plt.savefig("lifi_wavelength_comparison.png")
    plt.close()  # Закрываем фигуру после сохранения

    # Добавим сравнение лазеров с разными углами расходимости в градусах
    plt.figure(figsize=(10, 6))
    divergence_degrees = [0.5, 1, 5, 10, 20]
    for deg in divergence_degrees:
        tx = Transmitter(wavelength_850nm, 0.5, deg, 10, divergence_in_degrees=True)
        results = simulate_lifi_system(tx, base_receiver, distances)
        label = f"Угол расходимости = {deg}°"
        plt.plot(distances, results["snrs"], label=label)

    plt.title("Влияние угла расходимости луча на SNR")
    plt.xlabel("Расстояние (м)")
    plt.ylabel("SNR (дБ)")
    plt.grid(True)
    plt.legend()
    plt.savefig("lifi_beam_divergence_degrees_comparison.png")
    plt.close()  # Закрываем фигуру после сохранения

    print("Анализ завершен.")


# Дополнительная функция для анализа оптимальных параметров
def optimize_system_parameters(target_distance, target_ber=1e-6):
    """
    Функция для подбора оптимальных параметров системы для заданного расстояния и BER.

    Args:
        target_distance: Целевое расстояние в метрах
        target_ber: Целевой коэффициент ошибок (по умолчанию 10^-6)

    Returns:
        Словарь с оптимальными параметрами
    """
    # Диапазон параметров для поиска
    wavelengths = [850e-9, 1310e-9, 1550e-9]
    powers = np.linspace(0.1, 2.0, 20)  # От 100 мВт до 2 Вт
    beam_divergences = np.linspace(0.001, 0.01, 10)  # От 1 до 10 мрад
    bias_voltages = np.linspace(5, 20, 4)  # От 5В до 20В

    best_params = {
        "wavelength": None,
        "power": float("inf"),
        "beam_divergence": None,
        "bias_voltage": None,
        "snr": float("-inf"),
        "ber": float("inf"),
    }

    # Поиск оптимальных параметров
    for wavelength in wavelengths:
        for power in powers:
            for divergence in beam_divergences:
                for bias in bias_voltages:
                    # Создаем передатчик и приемник с текущими параметрами
                    tx = Transmitter(wavelength, power, divergence, 10)
                    rx = Receiver.from_photodiode_model("S5973", wavelength=wavelength)

                    # Симулируем для конкретного расстояния
                    power_density = tx.get_power_density(target_distance)
                    incident_power = power_density * rx.area
                    signal_current = rx.calculate_signal_current(incident_power)
                    noise_current = rx.calculate_noise_current()
                    snr = rx.calculate_snr(signal_current, noise_current)
                    ber = rx.calculate_ber(snr)

                    # Если BER соответствует требованиям и мощность меньше текущей лучшей
                    if ber <= target_ber and power < best_params["power"]:
                        best_params = {
                            "wavelength": wavelength,
                            "power": power,
                            "beam_divergence": divergence,
                            "bias_voltage": bias,
                            "snr": snr,
                            "ber": ber,
                        }

    # Преобразуем результаты для вывода
    if best_params["wavelength"] is not None:
        best_params["wavelength_nm"] = best_params["wavelength"] * 1e9
        best_params["power_mW"] = best_params["power"] * 1000
        best_params["beam_divergence_mrad"] = best_params["beam_divergence"] * 1000

    return best_params


# Функция для анализа системы LiFi для дрона
def analyze_drone_lifi_system():
    """
    Комплексный анализ системы LiFi для дрона на расстоянии до 300 метров.
    Исследуем оптимальные параметры передатчика и приемника.
    """
    print("=" * 80)
    print("АНАЛИЗ СИСТЕМЫ LiFi ДЛЯ ДРОНА (ДАЛЬНОСТЬ ДО 300 МЕТРОВ)")
    print("=" * 80)

    # Расстояния для анализа
    distances = np.linspace(0, 300, 1000)  # метры

    # Минимальный приемлемый SNR
    min_snr = 10  # дБ

    # 1. АНАЛИЗ МОЩНОСТИ ЛАЗЕРА/ИК-ДИОДА
    print("\n1. АНАЛИЗ ТРЕБУЕМОЙ МОЩНОСТИ ПЕРЕДАТЧИКА")
    print("-" * 50)

    powers = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 15.0, 65.0]  # Вт (от 10 мВт до 2 Вт)
    beam_divergences = [
        0.001,
        0.005,
        0.01,
        0.02,
        np.radians(20),
    ]  # радианы (0.057, 0.286, 0.573, 1.146 градусов)

    # Для более наглядного представления переведём в градусы
    beam_divergences_deg = [div * 180 / np.pi for div in beam_divergences]

    plt.figure(figsize=(12, 8))

    for divergence in beam_divergences:
        max_distances = []
        for power in powers:
            # Создаем передатчик и базовый приемник
            tx = Transmitter(850e-9, power, divergence, 10)
            rx = Receiver.from_photodiode_model("S5973")

            # Вычисляем максимальную дальность
            max_dist = estimate_max_distance(tx, rx, min_snr)
            max_distances.append(max_dist)

        # Переводим в градусы для меток
        div_deg = divergence * 180 / np.pi
        plt.plot(
            powers,
            max_distances,
            marker="o",
            label=f"Расходимость луча: {div_deg:.2f}°",
        )

    plt.title("Зависимость дальности связи от мощности передатчика")
    plt.xlabel("Мощность передатчика (Вт)")
    plt.ylabel("Максимальная дальность (м)")
    plt.grid(True)
    plt.legend()
    plt.savefig("lifi_drone_power_analysis.png")
    plt.close()  # Закрываем фигуру после сохранения

    # Детальный анализ для достижения 300 м
    print("Минимальная мощность для достижения дальности 300 м:")
    max_power = 145.0  # Вт, максимальная рассматриваемая мощность
    for divergence in beam_divergences:
        div_deg = divergence * 180 / np.pi
        for power in np.linspace(0.005, max_power, 500):
            tx = Transmitter(850e-9, power, divergence, 10)
            rx = Receiver.from_photodiode_model("S5973")
            max_dist = estimate_max_distance(tx, rx, min_snr)
            if max_dist >= 300:
                print(f"  Расходимость {div_deg:.2f}°: {power*1000:.1f} мВт")
                break
        else:
            print(
                f"  Расходимость {div_deg:.2f}°: > {max_power*1000:.1f} мВт (не достигается)"
            )

    # 2. АНАЛИЗ НАПРЯЖЕНИЯ СМЕЩЕНИЯ (VBIAS)
    print("\n2. АНАЛИЗ ВЛИЯНИЯ НАПРЯЖЕНИЯ СМЕЩЕНИЯ (VBIAS)")
    print("-" * 50)

    bias_voltages = [5, 10, 15, 20, 25, 30]  # В

    plt.figure(figsize=(12, 8))

    # Выбираем средние параметры передатчика
    tx_medium = Transmitter(850e-9, 0.5, 0.005, 10)

    for bias in bias_voltages:
        rx = Receiver.from_photodiode_model("S5973", bias_voltage=bias)
        results = simulate_lifi_system(tx_medium, rx, distances)
        plt.plot(distances, results["snrs"], label=f"VBias = {bias} В")

    plt.title("Влияние напряжения смещения на SNR")
    plt.xlabel("Расстояние (м)")
    plt.ylabel("SNR (дБ)")
    plt.grid(True)
    plt.legend()
    plt.axhline(
        y=min_snr, color="r", linestyle="--", label=f"Минимальный SNR ({min_snr} дБ)"
    )
    plt.savefig("lifi_drone_vbias_analysis.png")
    plt.close()  # Закрываем фигуру после сохранения

    print("Оптимальное напряжение смещения для достижения максимальной дальности:")
    for bias in bias_voltages:
        rx = Receiver.from_photodiode_model("S5973", bias_voltage=bias)
        max_dist = estimate_max_distance(tx_medium, rx, min_snr)
        print(f"  VBias = {bias} В: {max_dist:.1f} м")

    # 3. АНАЛИЗ ДЛИТЕЛЬНОСТИ ИМПУЛЬСОВ
    print("\n3. АНАЛИЗ ВЛИЯНИЯ ДЛИТЕЛЬНОСТИ ИМПУЛЬСОВ")
    print("-" * 50)

    pulse_durations = [3, 5, 10, 15]  # нс

    plt.figure(figsize=(12, 8))

    for duration in pulse_durations:
        tx = Transmitter(850e-9, 0.5, 0.005, duration)
        rx = Receiver.from_photodiode_model("S5973")

        results = simulate_lifi_system(tx, rx, distances)
        plt.plot(
            distances, results["snrs"], label=f"Длительность импульса = {duration} нс"
        )

    plt.title("Влияние длительности импульса на SNR")
    plt.xlabel("Расстояние (м)")
    plt.ylabel("SNR (дБ)")
    plt.grid(True)
    plt.legend()
    plt.axhline(
        y=min_snr, color="r", linestyle="--", label=f"Минимальный SNR ({min_snr} дБ)"
    )
    plt.savefig("lifi_drone_pulse_duration_analysis.png")
    plt.close()  # Закрываем фигуру после сохранения

    print("Влияние длительности импульса на максимальную дальность:")
    for duration in pulse_durations:
        tx = Transmitter(850e-9, 0.5, 0.005, duration)
        rx = Receiver.from_photodiode_model("S5973")

        max_dist = estimate_max_distance(tx, rx, min_snr)
        print(f"  Длительность импульса = {duration} нс: {max_dist:.1f} м")

    # 4. ВЛИЯНИЕ ПАРАМЕТРОВ ПРИЕМНИКА
    print("\n4. АНАЛИЗ КЛЮЧЕВЫХ ПАРАМЕТРОВ ПРИЕМНИКА")
    print("-" * 50)

    # 4.1 Площадь приемника
    areas = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]  # м² (от 0.1 до 10 см²)

    plt.figure(figsize=(12, 8))

    tx = Transmitter(850e-9, 0.5, 0.005, 10)

    for area in areas:
        rx = Receiver.from_photodiode_model("S5973", area=area)
        
        results = simulate_lifi_system(tx, rx, distances)
        plt.plot(
            distances, results["snrs"], label=f"Площадь приемника = {area*1e6:.1f} mm²"
        )

    plt.title("Влияние площади приемника на SNR")
    plt.xlabel("Расстояние (м)")
    plt.ylabel("SNR (дБ)")
    plt.grid(True)
    plt.legend()
    plt.axhline(
        y=min_snr, color="r", linestyle="--", label=f"Минимальный SNR ({min_snr} дБ)"
    )
    plt.savefig("lifi_drone_receiver_area_analysis.png")
    plt.close()  # Закрываем фигуру после сохранения

    print("Влияние площади приемника на максимальную дальность:")
    for area in areas:
        rx = Receiver.from_photodiode_model("S5973", area=area)
        max_dist = estimate_max_distance(tx, rx, min_snr)
        print(f"  Площадь приемника = {area*1e4:.1f} см²: {max_dist:.1f} м")

    # 4.2 Чувствительность приемника
    sensitivities = [0.3, 0.5, 0.7, 0.9]  # А/Вт

    plt.figure(figsize=(12, 8))

    for sens in sensitivities:
        rx = Receiver.from_photodiode_model("S5973", sensitivity=sens)
        
        results = simulate_lifi_system(tx, rx, distances)
        plt.plot(
            distances, results["snrs"], label=f"Чувствительность = {sens:.1f} А/Вт"
        )

    plt.title("Влияние чувствительности приемника на SNR")
    plt.xlabel("Расстояние (м)")
    plt.ylabel("SNR (дБ)")
    plt.grid(True)
    plt.legend()
    plt.axhline(
        y=min_snr, color="r", linestyle="--", label=f"Минимальный SNR ({min_snr} дБ)"
    )
    plt.savefig("lifi_drone_sensitivity_analysis.png")
    plt.close()  # Закрываем фигуру после сохранения

    # 4.3 Уровень шума
    noise_densities = [1e-13, 5e-13, 1e-12, 5e-12, 1e-11]  # W/Hz

    plt.figure(figsize=(12, 8))

    for noise in noise_densities:
        rx = Receiver.from_photodiode_model("S5973", noise_density=noise)

        results = simulate_lifi_system(tx, rx, distances)
        plt.plot(
            distances,
            results["snrs"],
            label=f"Шумовая плотность = {noise:.1e} W/Hz",
        )

    plt.title("Влияние уровня шума приемника на SNR")
    plt.xlabel("Расстояние (м)")
    plt.ylabel("SNR (дБ)")
    plt.grid(True)
    plt.legend()
    plt.axhline(
        y=min_snr, color="r", linestyle="--", label=f"Минимальный SNR ({min_snr} дБ)"
    )
    plt.savefig("lifi_drone_noise_analysis.png")
    plt.close()  # Закрываем фигуру после сохранения

    # 5. ОПТИМАЛЬНЫЕ ПАРАМЕТРЫ ДЛЯ ДАЛЬНОСТИ 300 м
    print("\n5. ОПТИМАЛЬНЫЕ ПАРАМЕТРЫ ДЛЯ ДАЛЬНОСТИ 300 м")
    print("-" * 50)

    # Комбинация мощности и площади приемника
    plt.figure(figsize=(12, 8))

    powers_300m = [0.5, 1.0, 2.0]
    areas_300m = [5e-5, 1e-4, 5e-4, 1e-3]  # м² (0.5, 1, 5, 10 см²)

    for power in powers_300m:
        max_distances = []
        for area in areas_300m:
            tx = Transmitter(850e-9, power, 0.005, 10)
            rx = Receiver.from_photodiode_model("S5973", area=area)

            max_dist = estimate_max_distance(tx, rx, min_snr)
            max_distances.append(max_dist)

        plt.plot(
            [area * 1e4 for area in areas_300m],
            max_distances,
            marker="o",
            label=f"Мощность = {power*1000:.0f} мВт",
        )

    plt.title(
        "Комбинация мощности передатчика и площади приемника для достижения требуемой дальности"
    )
    plt.xlabel("Площадь приемника (см²)")
    plt.ylabel("Максимальная дальность (м)")
    plt.grid(True)
    plt.legend()
    plt.axhline(y=300, color="r", linestyle="--", label="Целевая дальность (300 м)")
    plt.savefig("lifi_drone_optimal_params.png")
    plt.close()  # Закрываем фигуру после сохранения

    print("Рекомендуемые параметры системы для достижения дальности 300 м:")
    print("  1. Мощность лазера/ИК-диода: 1-2 Вт")
    print("  2. Угол расходимости луча: не более 0.5° (0.01 рад)")
    print("  3. Напряжение смещения (VBias): 15-25 В")
    print("  4. Площадь приемника: от 1 см²")
    print("  5. Чувствительность приемника: не менее 0.6 А/Вт")
    print("  6. Уровень шума: не более 1e-12 А/Гц^0.5")

    print("\nПри выборе приемного диода необходимо обратить внимание на:")
    print("  1. Чувствительность - чем выше, тем лучше")
    print("  2. Низкий уровень шума")
    print("  3. Высокую квантовую эффективность")
    print("  4. Максимальное напряжение смещения - чем выше, тем лучше")
    print(
        "  5. Быстродействие (время нарастания и спада) - для работы с короткими импульсами"
    )
    print("  6. Широкий динамический диапазон")
    print("  7. Устойчивость к внешним световым помехам")


# Функция для анализа высокомощного лазера с большой расходимостью
def analyze_high_power_wide_angle_laser(corner_case=20):
    """
    Анализ производительности лазера с высокой мощностью (15-145 Вт)
    и большим углом расходимости (20 градусов)
    """
    print("\n" + "=" * 80)
    print(f"АНАЛИЗ ЛАЗЕРА С РАСХОДИМОСТЬЮ {corner_case}° И МОЩНОСТЬЮ 15-145 Вт")
    print("=" * 80)

    # Расстояния для анализа
    distances = np.linspace(0, 300, 1000)  # метры

    # Мощности лазера для анализа
    powers = [15, 30, 65, 145]  # Вт

    # Параметры приемников разной площади
    receiver_areas = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]  # м² (1, 5, 10, 20, 50 см²)


    # Минимальный приемлемый SNR
    min_snr = 10  # дБ

    # 1. АНАЛИЗ ВЛИЯНИЯ МОЩНОСТИ НА ДАЛЬНОСТЬ
    plt.figure(figsize=(12, 8))

    base_receiver = Receiver.from_photodiode_model("S5973")

    for power in powers:
        laser = Transmitter(850e-9, power, corner_case, 10, divergence_in_degrees=True)
        results = simulate_lifi_system(laser, base_receiver, distances)
        label = f"{power} Вт, {corner_case}°"
        plt.plot(distances, results["snrs"], label=label)

    plt.title(f"Влияние мощности лазера ({corner_case}°) на SNR")
    plt.xlabel("Расстояние (м)")
    plt.ylabel("SNR (дБ)")
    plt.grid(True)
    plt.legend()
    plt.axhline(
        y=min_snr, color="r", linestyle="--", label=f"Минимальный SNR ({min_snr} дБ)"
    )
    plt.savefig("lifi_high_power_laser_analysis.png")
    plt.close()  # Закрываем фигуру после сохранения

    # Расчет максимальной дальности для разных мощностей
    print(f"\nМаксимальная дальность для лазера с углом {corner_case} градусов:")
    for power in powers:
        laser = Transmitter(850e-9, power, corner_case, 10, divergence_in_degrees=True)
        max_dist = estimate_max_distance(laser, base_receiver, min_snr)
        print(f"  При мощности {power} Вт: {max_dist:.1f} м")

    # 2. АНАЛИЗ ВЛИЯНИЯ ПЛОЩАДИ ПРИЕМНИКА
    plt.figure(figsize=(12, 8))

    # Для разных площадей приемника с фиксированной мощностью
    power = 60  # Вт
    laser = Transmitter(850e-9, power, corner_case, 10, divergence_in_degrees=True)

    for area in receiver_areas:
        rx = Receiver.from_photodiode_model("S5973", area=area)

        results = simulate_lifi_system(laser, rx, distances)
        label = f"Площадь приемника: {area*1e4:.1f} см²"
        plt.plot(distances, results["snrs"], label=label)

    plt.title(f"Влияние площади приемника на SNR (лазер {power} Вт, {corner_case}°)")
    plt.xlabel("Расстояние (м)")
    plt.ylabel("SNR (дБ)")
    plt.grid(True)
    plt.legend()
    plt.axhline(
        y=min_snr, color="r", linestyle="--", label=f"Минимальный SNR ({min_snr} дБ)"
    )
    plt.savefig("lifi_high_power_laser_receiver_area.png")
    plt.close()  # Закрываем фигуру после сохранения

    # 3. ВЛИЯНИЕ НАПРЯЖЕНИЯ СМЕЩЕНИЯ (VBIAS)
    plt.figure(figsize=(12, 8))

    bias_voltages = [5, 10, 15, 20, 25, 30]  # В

    for bias in bias_voltages:
        rx = Receiver.from_photodiode_model("S5973", bias_voltage=bias)

        results = simulate_lifi_system(laser, rx, distances)
        plt.plot(distances, results["snrs"], label=f"VBias = {bias} В")

    plt.title(f"Влияние напряжения смещения на SNR (лазер {power} Вт, {corner_case}°)")
    plt.xlabel("Расстояние (м)")
    plt.ylabel("SNR (дБ)")
    plt.grid(True)
    plt.legend()
    plt.axhline(
        y=min_snr, color="r", linestyle="--", label=f"Минимальный SNR ({min_snr} дБ)"
    )
    plt.savefig("lifi_high_power_laser_vbias.png")
    plt.close()  # Закрываем фигуру после сохранения

    # 4. ОПТИМАЛЬНЫЕ КОМБИНАЦИИ ПАРАМЕТРОВ ДЛЯ ДОСТИЖЕНИЯ 300 М
    print("\nОптимальные комбинации параметров для достижения дальности 300 м:")
    print(f"  Лазер: расходимость {corner_case}°")

    target_distance = 300  # м

    for power in powers:
        for area in receiver_areas:
            for bias in [15, 30]:  # В
                laser = Transmitter(850e-9, power, corner_case, 10, divergence_in_degrees=True)
                rx = Receiver.from_photodiode_model("S5973", area=area, bias_voltage=bias)

                # Получаем SNR на целевом расстоянии
                power_density = laser.get_power_density(target_distance)
                incident_power = power_density * rx.area
                signal_current = rx.calculate_signal_current(incident_power)
                noise_current = rx.calculate_noise_current()
                snr = rx.calculate_snr(signal_current, noise_current)

                if snr >= min_snr:
                    print(
                        f"  - Мощность: {power} Вт, Площадь приемника: {area*1e4:.1f} см², VBias: {bias} В (SNR = {snr:.1f} дБ)"
                    )

    # 5. АНАЛИЗ ДЛИТЕЛЬНОСТИ ИМПУЛЬСОВ
    plt.figure(figsize=(12, 8))

    pulse_durations = [3, 5, 10, 15]  # нс
    power = 60  # Вт

    rx = Receiver.from_photodiode_model("S5973")

    for duration in pulse_durations:
        laser = Transmitter(850e-9, power, corner_case, duration, divergence_in_degrees=True)
        results = simulate_lifi_system(laser, rx, distances)
        plt.plot(
            distances, results["snrs"], label=f"Длительность импульса = {duration} нс"
        )

    plt.title(f"Влияние длительности импульса на SNR (лазер {power} Вт, {corner_case}°)")
    plt.xlabel("Расстояние (м)")
    plt.ylabel("SNR (дБ)")
    plt.grid(True)
    plt.legend()
    plt.axhline(
        y=min_snr, color="r", linestyle="--", label=f"Минимальный SNR ({min_snr} дБ)"
    )
    plt.savefig("lifi_high_power_laser_pulse_duration.png")
    plt.close()  # Закрываем фигуру после сохранения

    # 6. ИТОГОВЫЕ РЕКОМЕНДАЦИИ
    print(f"\nРЕКОМЕНДАЦИИ ДЛЯ ИСПОЛЬЗОВАНИЯ ЛАЗЕРА С РАСХОДИМОСТЬЮ {corner_case}°:")
    print("1. Для достижения дальности связи 300 м при данной расходимости:")
    print("   - Требуется высокая мощность лазера (60+ Вт)")
    print("   - Рекомендуется использовать приемник с площадью не менее 10 см²")
    print("   - Рекомендуемое напряжение смещения (VBias): 15-30 В")
    print("2. Оптимальные параметры приемника:")
    print("   - Высокая чувствительность (0.6+ А/Вт)")
    print("   - Высокая квантовая эффективность (> 0.7)")
    print("   - Низкий темновой ток и уровень шума")
    print("   - Быстродействие для работы с короткими импульсами (3-15 нс)")
    print("3. Компромисс между мощностью и площадью приемника:")
    print(
        "   - При использовании мощности 145 Вт можно обойтись меньшей площадью приемника"
    )
    print(
        "   - При ограничении мощности до 30-60 Вт необходимо увеличить площадь приемника до 20-50 см²"
    )
    print(f"\nПРИМЕЧАНИЕ: Из-за большого угла расходимости ({corner_case}°) потребуется значительно"
    )
    print("более высокая мощность лазера по сравнению с узконаправленными системами.")


# Функция для анализа высоких отрицательных значений Vbias
def analyze_high_negative_vbias():
    """
    Анализ производительности системы LiFi при использовании
    высоких отрицательных напряжений смещения (до -300 В)
    """
    print("\n" + "=" * 80)
    print("АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ С ВЫСОКИМИ ОТРИЦАТЕЛЬНЫМИ VBIAS (ДО -300 В)")
    print("=" * 80)

    # Расстояния для анализа
    distances = np.linspace(0, 300, 1000)  # метры

    # Различные напряжения смещения для анализа
    bias_voltages = [-15, -30, -50, -100, -150, -200, -250, -300]  # В

    # Параметры приемника
    diode_model = "S5973"  
    sensitivity = PHOTODIODE_PRESETS[diode_model]["sensitivity"]  # А/Вт


    # Минимальный приемлемый SNR
    min_snr = 10  # дБ

    # Используем модель для масштабирования чувствительности при увеличении VBias
    # Типичная модель: чувствительность растёт до определенного предела с увеличением VBias
    def scaled_sensitivity(vbias):
        # Простая модель: логарифмический рост с выходом на плато
        base_sensitivity = sensitivity
        max_gain = 2.5  # максимальное увеличение чувствительности
        saturation_voltage = 150  # напряжение, после которого рост замедляется

        # Используем модуль напряжения для расчета
        abs_vbias = abs(vbias)

        if abs_vbias <= 30:
            # Линейный рост на низких напряжениях
            gain = 1.0 + (abs_vbias - 15) / 15 * 0.5
        else:
            # Логарифмический рост и выход на плато
            gain = 1.5 + (max_gain - 1.5) * (
                1 - np.exp(-(abs_vbias - 30) / saturation_voltage)
            )

        return base_sensitivity * min(gain, max_gain)

    # Параметры для анализа
    wavelengths = [850e-9, 1550e-9]  # м
    power = 1.0  # Вт
    beam_divergence = 0.005  # радианы (~0.3 градуса)
    pulse_duration = 10  # нс

    # 1. АНАЛИЗ ВЛИЯНИЯ VBIAS НА ЧУВСТВИТЕЛЬНОСТЬ И ДАЛЬНОСТЬ
    sensitivities = [scaled_sensitivity(bias) for bias in bias_voltages]

    plt.figure(figsize=(10, 6))
    plt.plot(np.abs(bias_voltages), sensitivities, marker="o")
    plt.title("Зависимость чувствительности приемника от напряжения смещения")
    plt.xlabel("Абсолютное значение VBias (В)")
    plt.ylabel("Чувствительность (А/Вт)")
    plt.grid(True)
    plt.savefig("lifi_high_vbias_sensitivity.png")
    plt.close()  # Закрываем фигуру после сохранения

    # 2. АНАЛИЗ ВЛИЯНИЯ VBIAS НА SNR
    plt.figure(figsize=(12, 8))

    transmitter = Transmitter(850e-9, power, beam_divergence, pulse_duration)

    for bias in bias_voltages:
        # Создаем приемник с масштабированной чувствительностью
        rx = Receiver.from_photodiode_model("S5973", sensitivity = scaled_sensitivity(bias), bias_voltage=abs(bias))

        results = simulate_lifi_system(transmitter, rx, distances)
        plt.plot(distances, results["snrs"], label=f"VBias = {bias} В")

    plt.title("Влияние высоких отрицательных значений VBias на SNR")
    plt.xlabel("Расстояние (м)")
    plt.ylabel("SNR (дБ)")
    plt.grid(True)
    plt.legend()
    plt.axhline(
        y=min_snr, color="r", linestyle="--", label=f"Минимальный SNR ({min_snr} дБ)"
    )
    plt.savefig("lifi_high_negative_vbias_snr.png")
    plt.close()  # Закрываем фигуру после сохранения

    # 3. МАКСИМАЛЬНАЯ ДАЛЬНОСТЬ ПРИ РАЗНЫХ VBIAS
    max_distances = []

    for bias in bias_voltages:
        rx = Receiver.from_photodiode_model("S5973", sensitivity=scaled_sensitivity(bias), bias_voltage=abs(bias))

        max_dist = estimate_max_distance(transmitter, rx, min_snr)
        max_distances.append(max_dist)
        print(
            f"  VBias = {bias} В: Максимальная дальность {max_dist:.1f} м, Чувствительность {scaled_sensitivity(bias):.2f} А/Вт"
        )

    plt.figure(figsize=(10, 6))
    plt.plot(np.abs(bias_voltages), max_distances, marker="o")
    plt.title("Зависимость максимальной дальности от напряжения смещения")
    plt.xlabel("Абсолютное значение VBias (В)")
    plt.ylabel("Максимальная дальность (м)")
    plt.grid(True)
    plt.savefig("lifi_high_vbias_max_distance.png")
    plt.close()  # Закрываем фигуру после сохранения

    # 4. КОМБИНИРОВАННЫЙ АНАЛИЗ: ВЫСОКОЕ VBIAS + ШИРОКОУГОЛЬНЫЙ ЛАЗЕР
    print("\nАнализ влияния высокого VBias на работу с широкоугольными лазерами:")

    # Лазер с большим углом расходимости (25 градусов)
    wide_angle_laser = Transmitter(850e-9, 60, 25, 10, divergence_in_degrees=True)

    # Сравниваем стандартное и высокое отрицательное напряжение
    plt.figure(figsize=(12, 8))

    # Стандартное напряжение
    rx_standard = Receiver.from_photodiode_model("S5973")

    results = simulate_lifi_system(wide_angle_laser, rx_standard, distances)
    plt.plot(distances, results["snrs"], label=f"VBias = -15 В (стандартный)")

    # Высокое отрицательное напряжение
    for bias in [-100, -200, -300]:
        rx_high = Receiver.from_photodiode_model("S5973",sensitivity =scaled_sensitivity(bias),  bias_voltage=abs(bias))

        results = simulate_lifi_system(wide_angle_laser, rx_high, distances)
        plt.plot(distances, results["snrs"], label=f"VBias = {bias} В")

    plt.title("Влияние высоких значений VBias на работу с широкоугольным лазером (25°)")
    plt.xlabel("Расстояние (м)")
    plt.ylabel("SNR (дБ)")
    plt.grid(True)
    plt.legend()
    plt.axhline(
        y=min_snr, color="r", linestyle="--", label=f"Минимальный SNR ({min_snr} дБ)"
    )
    plt.savefig("lifi_high_vbias_wide_angle_laser.png")
    plt.close()  # Закрываем фигуру после сохранения

    # 5. ВЛИЯНИЕ ВЫСОКОГО VBIAS НА СКОРОСТЬ ОТКЛИКА
    print("\nВлияние высокого VBias на быстродействие приемника:")
    print(
        "  Высокое отрицательное напряжение смещения значительно улучшает быстродействие"
    )
    print("  фотодиодов, что позволяет работать с более короткими импульсами.")

    # Оценка пропускной способности для разных VBias
    response_times = []  # нс
    bandwidths_mhz = []  # МГц

    for bias in bias_voltages:
        # Примерная модель: время отклика уменьшается с увеличением напряжения
        # до определенного предела
        abs_bias = abs(bias)
        base_response = 10  # нс при низком напряжении
        min_response = 0.5  # нс минимально возможное время отклика

        # Упрощенная формула зависимости времени отклика от напряжения
        response_time = base_response * np.exp(-abs_bias / 100) + min_response
        response_times.append(response_time)

        # Оценка пропускной способности
        bandwidth_mhz = 350 / response_time  # МГц
        bandwidths_mhz.append(bandwidth_mhz)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = "tab:blue"
    ax1.set_xlabel("Абсолютное значение VBias (В)")
    ax1.set_ylabel("Время отклика (нс)", color=color)
    ax1.plot(np.abs(bias_voltages), response_times, marker="o", color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Пропускная способность (МГц)", color=color)
    ax2.plot(np.abs(bias_voltages), bandwidths_mhz, marker="s", color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Влияние VBias на быстродействие приемника")
    plt.grid(True)
    plt.savefig("lifi_high_vbias_response_time.png")
    plt.close()  # Закрываем фигуру после сохранения

    # 6. ИТОГОВЫЕ РЕКОМЕНДАЦИИ
    print("\nРЕКОМЕНДАЦИИ ПО ИСПОЛЬЗОВАНИЮ ВЫСОКОГО ОТРИЦАТЕЛЬНОГО VBIAS:")
    print("1. Оптимальный диапазон напряжений: от -150 до -300 В")
    print("   - Значительный прирост чувствительности до ~-150 В")
    print("   - Дополнительные улучшения в диапазоне -150...-300 В")
    print("2. Преимущества высокого VBias:")
    print("   - Увеличение чувствительности до 2.5 раз")
    print("   - Увеличение максимальной дальности связи на 60-80%")
    print("   - Снижение времени отклика до <1 нс")
    print("   - Увеличение пропускной способности до >300 МГц")
    print("3. Особенности использования с широкоугольными лазерами:")
    print(
        "   - С VBias = -300 В возможно достижение дальности 300 м даже с лазером 25°"
    )
    print(
        "   - Комбинация высокого VBias и высокой мощности лазера дает наилучшие результаты"
    )
    print("4. Требования к компонентам системы:")
    print("   - Выбор фотодиодов, поддерживающих высокие обратные напряжения")
    print("   - Необходимость теплоотвода из-за повышенной рассеиваемой мощности")
    print("   - Экранирование для предотвращения электромагнитных помех")
    print("   - Соблюдение мер электробезопасности")


# Добавляем вызов новой функции в основной блок
if __name__ == "__main__":
    # Расстояния для анализа
    distances = np.linspace(0, 300, 1000)  # метры
    divergence = 14
    # Базовая симуляция
    # main()

    # Анализ лазера с большим углом расходимости (14 градусов)
    analyze_high_divergence_laser(distances, "S5973", divergence)

    # Анализ системы LiFi для дрона
    # analyze_drone_lifi_system()

    # Анализ высокомощного лазера с большой расходимостью (20°)
    # analyze_high_power_wide_angle_laser(divergence)

    # Анализ производительности с высокими отрицательными VBias
    analyze_high_negative_vbias()





