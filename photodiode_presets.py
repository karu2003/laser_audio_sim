# Словарь с параметрами известных фотодиодов
PHOTODIODE_PRESETS = {
    "S5973": {
        "type": "Si",  # Тип фотодиода
        "area": 0.12e-6,  # м² (0.12 мм²)
        "sensitivity": 0.51,  # А/Вт
        "quantum_efficiency": 0.7,
        "dark_current": 0.001e-9,  # А (0.001 нА)
        "noise": "NEP",  # W/Hz Noise Equivalent Power
        "NEP": 1.1e-15, 
        "max_bias_voltage": 20,  # В
        "terminal_capacitance": 1.6e-12,  # Ф (1.6 пФ)
        "bandwidth": 1e9,  # 1GHz
        "wavelength_range": (320e-9, 1100e-9),  # м (320-1100 нм)
        "peak_wavelength": 760e-9,  # м (850 нм)
    },
    "S5971": {
        "type": "Si",  # Тип фотодиода
        "area": 1.1e-6,  # м² (1 мм²)
        "sensitivity": 0.55,  # А/Вт
        "quantum_efficiency": 0.65,
        "dark_current": 0.07e-9,  # А (0.2 нА)
        "noise": "NEP",  # W/Hz Noise Equivalent Power 
        "NEP": 7.4e-15,  # W/Hz
        "max_bias_voltage": 20,  # В
        "terminal_capacitance": 3e-12,  # Ф (10 пФ)
        "bandwidth": 0.1e9,
        "wavelength_range": (320e-9, 1100e-9),  # м (320-1100 нм)
        "peak_wavelength": 920e-9,  # м (850 нм)
    },
    "G8931-04": {
        "type": "APD",  # Тип фотодиода
        "area": 1.256e-9,  # м² (0.001256 мм² - 0.04 mm)
        "sensitivity": 0.9,  # А/Вт
        "quantum_efficiency": 0.8,
        "dark_current": 40e-9,  # А (40 нА)
        "noise": "ENF", # Excess Noise Figure
        "ENF": 0.7, 
        "max_bias_voltage": 55,  # В
        "terminal_capacitance": 2e-12,  # Ф (2 пФ)
        "bandwidth": 4e9,
        "wavelength_range": (900e-9, 1700e-9),  # м (900-1700 нм)
        "peak_wavelength": 1550e-9,  # м (1550 нм)
    },
    "G8931-10": {
        "type": "APD",  # Тип фотодиода
        "area": 7.85e-9,  # м² (0.00785 мм² - 0.1 mm)
        "sensitivity": 0.9,  # А/Вт
        "quantum_efficiency": 0.8,
        "dark_current": 90e-9,  # А (90 нА)
        "noise": "ENF", # Excess Noise Figure
        "ENF": 0.7, 
        "max_bias_voltage": 55,  # В
        "terminal_capacitance": 2e-12,  # Ф (2 пФ)
        "bandwidth": 1.5e9,
        "wavelength_range": (900e-9, 1700e-9),  # м (900-1700 нм)
        "peak_wavelength": 1550e-9,  # м (1550 нм)
    },
     "G8931-20": {
        "type": "APD",  # Тип фотодиода
        "area": 3.14e-8,  # м² (0.0314 мм² -0.2 mm)
        "sensitivity": 0.9,  # А/Вт
        "quantum_efficiency": 0.8,
        "dark_current": 150e-9,  # А (150 нА)
        "noise": "ENF", # Excess Noise Figure
        "ENF": 0.7, 
        "max_bias_voltage": 55,  # В
        "terminal_capacitance": 2e-12,  # Ф (2 пФ)
        "bandwidth": 0.9e9,
        "wavelength_range": (900e-9, 1700e-9),  # м (900-1700 нм)
        "peak_wavelength": 1550e-9,  # м (1550 нм)
    },
    
    # Кремниевые фотодиоды PIN
    "BPW34": {
        "type": "Si",  # Тип фотодиода
        "area": 7.5e-6,  # м² (7.5 мм²)
        "sensitivity": 0.62,  # А/Вт при 850 нм
        "quantum_efficiency": 0.75,
        "dark_current": 2e-9,  # А (2 нА)
        "noise": "NEP", # Excess Noise Figure
        "NEP": 4.e-14,  # W/Hz
        "max_bias_voltage": 60,  # В
        "terminal_capacitance": 25e-12,  # Ф (25 пФ)
        "bandwidth": 3.49e6,  # Гц (3.49 МГц)
        "wavelength_range": (350e-9, 1100e-9),  # м (350-1100 нм)
        "peak_wavelength": 900e-9,  # м (900 нм)
    },
    "VEMD8081": {
        "type": "Si",  # Тип фотодиода
        "area": 5.4e-6,  # м² (7.5 мм²)
        "sensitivity": 0.62,  # А/Вт при 850 нм
        "quantum_efficiency": 0.75,
        "dark_current": 0.5e-9,  # А (2 нА)
        "noise": "NEP", # Excess Noise Figure
        "NEP": 4.e-14,  # W/Hz
        "max_bias_voltage": 20,  # В
        "terminal_capacitance": 20e-12,  # Ф (20 пФ)
        "bandwidth": 9e6,  # Гц (9 МГц)
        "wavelength_range": (350e-9, 1100e-9),  # м (350-1100 нм)
        "peak_wavelength": 840e-9,  # м (840 нм)
    },
    # Лавинные фотодиоды (APD)
    "C30902": {
        "type": "APD",  # Тип фотодиода
        "area": 0.2e-6,  # м² (0.2 мм²)
        "sensitivity": 70,  # А/Вт при максимальном усилении
        "quantum_efficiency": 0.85,
        "dark_current": 7e-9,  # А (7 нА)
        "noise": "NEP", # Excess Noise Figure
        "NEP": 1.23e-15,  # W/Hz
        "max_bias_voltage": 225,  # В
        "terminal_capacitance": 1.5e-12,  # Ф (1.5 пФ)
        "bandwidth": 0.8e9, # Гц (800 МГц)
        "wavelength_range": (400e-9, 1100e-9),  # м (400-1100 нм)
        "peak_wavelength": 800e-9,  # м (800 нм)
    },
    # InGaAs фотодиоды для 1550 нм
    "IG26X250S41": {
        "type": "InGaAs",  # Тип фотодиода
        "area": 4.90625e-8,  # м² (0.25 mm)
        "sensitivity": 1.5,  # А/Вт
        "quantum_efficiency": 0.75,
        "dark_current": 2e-6,  # А (2 uА)
        "noise": "NEP", # Excess Noise Figure
        "NEP": 4.2e-13,  # W/Hz
        "max_bias_voltage": 5,  # В
        "terminal_capacitance": 35e-12,  # Ф (35 пФ)
        "bandwidth": 2.e6,  # Гц (2 МГц)
        "wavelength_range": (900e-9, 1700e-9),  # м (900-1700 нм)
        "peak_wavelength": 1550e-9,  # м (1550 нм)
    },
}