def calculate_nep(q, Id, F, R):
    """
    Расчет NEP (Noise Equivalent Power) по формуле:
    NEP = sqrt(2 * q * Id * F) / R
    """
    nep = (2 * q * Id * F) ** 0.5 / R
    return nep

# Константы
q = 1.6e-19      # заряд электрона (Кл)
Id = 65e-9       # темновой ток (А)
F = 0.7          # избыточный шумовой фактор
R = 0.9          # чувствительность (A/W)

# Расчет NEP
nep_result = calculate_nep(q, Id, F, R)
print(f"NEP для G8931-04: {nep_result:.2e} Вт/√Гц")
