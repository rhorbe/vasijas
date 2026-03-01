#%%
# ids_a_renombrar = {
#     'Al_03G': 'AL_03G',
#     'Al_06A': 'Al_06A', 'BA115_5': None, 'BA132_1': None, 'BO3_2_1': None, 'BO_10_1': None,
#     'BO_10_1plato': None, 'BO_11a_1': None, 'BO_11a_2': None, 'BO_11a_3': None, 'BO_11b_1': None, 'BO_14_1': None,
#     'BO_14_2': None, 'BO_16_1': None, 'BO_17_1': None, 'BO_18_1': None, 'BO_18_2': None, 'BO_19_1': None,
#     'BO_19_2': None, 'BO_20_1': None, 'BO_21_1': None, 'BO_21_2': None, 'BO_22_1': None, 'BO_23_1': None,
#     'BO_24_1': None, 'BO_26_1': None, 'BO_27_1': None, 'BO_27_2': None, 'BO_28_1': None, 'BO_28_2': None,
#     'BO_29_1': None, 'BO_29_2': None, 'BO_29_3': None, 'BO_29_4': None, 'BO_29_5': None, 'BO_30_1': None,
#     'BO_30_2': None, 'BO_31_1': None, 'BO_31_2': None, 'BO_31_3': None, 'BO_31_4': None, 'BO_3_1': None,
#     'BO_4_1': None, 'BO_4_2': None, 'BO_4_3': None, 'BO_5_1': None, 'BO_5_1plato': None, 'BO_6_1': None,
#     'BO_6_2': None, 'BO_7_1': None, 'BO_7_2': None, 'BO_9_1': None, 'BO_9_2': None, 'BO_9_3': None, 'BO_9_4': None,
#     'CC0_8_2': None, 'CC_11_14': None, 'CC_11_15': None, 'CC_11_17': None, 'CC_11_18': None, 'CC_13_14_1_3': None,
#     'CC_13_14_2_1': None, 'CC_13_14_2_3': None, 'ER_23_1': None, 'PN10_1': None
# }

# archivos_a_renombrar = {
#     'CC_13_14 1_3': 'CC_13_14_1_3',
#     'CC_13_14 2_1': 'CC_13_14_2_1',
#     'CC_13_14 2_3': 'CC_13_14_2_3',
#     'ER_23-1': 'ER_23_1'
ubicaciones = [
    (item.name, item)
    for item in PATHS.DATA_ORIGINALES.iterdir()
    if item.is_dir()
]

for ubicacion, ruta_ubicacion in ubicaciones:
    ruta_destino = PATHS.DATA_CORREGIDOS / ubicacion
    ruta_destino.mkdir(parents=True, exist_ok=True)

    archivos_png = ruta_ubicacion.glob("*.png")

    for archivo in archivos_png:
        nombre_normalizado = normalizar_nombre_archivo(archivo.name)
        destino = ruta_destino / nombre_normalizado

        if archivo.name != nombre_normalizado:
            archivo.rename(destino)
# }



ruta_ubicaciones = [
    ruta
    for ruta in PATHS.DATA_ORIGINALES.iterdir()
    if ruta.is_dir()
]

for ruta_ubicacion in ruta_ubicaciones:
    nombre_ubicacion = ruta_ubicacion.name

    ruta_destino = PATHS.DATA_CORREGIDOS / nombre_ubicacion
    ruta_destino.mkdir(parents=True, exist_ok=True)

    archivos_png = ruta_ubicacion.glob("*.png")
    print(ruta_ubicacion)

    for archivo in archivos_png:
        nombre_normalizado = normalizar_nombre_archivo(archivo.name)
        destino = ruta_destino / nombre_normalizado
        archivo.rename(destino)