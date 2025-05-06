def calculate_nep(q, Id, F, R):
    """
    Расчет NEP (Noise Equivalent Power) по формуле:
    NEP = sqrt(2 * q * Id * F) / R
    """
    nep = (2 * q * Id * F) ** 0.5 / R
    return nep

def calculate_nep(noise_current_a_per_sqrtHz, responsivity_a_per_w):
    """
    Расчёт NEP (Noise Equivalent Power)
    
    Parameters:
        noise_current_a_per_sqrtHz: токовый шум в А/√Гц
        responsivity_a_per_w: чувствительность в А/Вт
        
    Returns:
        NEP в Вт/√Гц
    """
    if responsivity_a_per_w == 0:
        raise ValueError("Responsivity cannot be zero.")
    return noise_current_a_per_sqrtHz / responsivity_a_per_w


def calculate_bandwidth(t_rise_seconds):
    """
    Расчёт полосы пропускания по времени нарастания (формула BW = 0.35 / tr)
    
    Parameters:
        t_rise_seconds: время нарастания в секундах
        
    Returns:
        полоса пропускания в Гц
    """
    if t_rise_seconds <= 0:
        raise ValueError("Rise time must be positive.")
    return 0.35 / t_rise_seconds


# Константы
q = 1.6e-19      # заряд электрона (Кл)
Id = 65e-9       # темновой ток (А)
F = 0.7          # избыточный шумовой фактор
R = 0.9          # чувствительность (A/W)

# Расчет NEP
nep_result = calculate_nep(q, Id, F, R)
print(f"NEP для G8931-04: {nep_result:.2e} Вт/√Гц")

noise_current = 1e-12        # 1 пА/√Гц
responsivity = 0.55          # A/W
t_rise = 600e-12             # 600 пс

nep = calculate_nep(noise_current, responsivity)
bw = calculate_bandwidth(t_rise)

print(f"NEP: {nep:.2e} W/√Hz")
print(f"Bandwidth: {bw/1e6:.2f} MHz")
