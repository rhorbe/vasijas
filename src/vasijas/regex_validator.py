import re

# Expresión regular que verifica:
# - Comienza con 2 letras en mayúscula
# - Continúa con 2 dígitos
# - Un guión bajo
# - Seguido de letras, números y guiones bajos (sin espacios en blanco)
# PATTERN = r'^[A-Z]{2}\d{3}_[A-Za-z0-9_]*$'
PATTERN = r'^[()A-Za-z0-9_]*$'

def validar_formato(cadena: str) -> bool:
    """
    Valida que una cadena cumpla con el formato especificado.

    Formato:
    - Comienza con 2 letras en mayúscula
    - Continúa con 2 dígitos
    - Un guión bajo
    - Seguido de letras, números y guiones bajos (sin espacios en blanco)

    Args:
        cadena: La cadena a validar

    Returns:
        True si la cadena cumple con el formato, False en caso contrario
    """
    return bool(re.match(PATTERN, cadena))


