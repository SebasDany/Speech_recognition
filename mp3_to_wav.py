from os import path
from pydub import AudioSegment

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as waves
def graphSpectr(file):
    # archivo = input('archivo de sonido:' )
    archivo = file
    muestreo, sonido = waves.read(archivo)

    # canales: monofónico o estéreo
    tamano = np.shape(sonido)
    muestras = tamano[0]
    m = len(tamano)
    canales = 1  # monofónico
    if (m > 1):  # estéreo
        canales = tamano[1]
    # experimento con un canal
    if (canales > 1):
        canal = 0
        uncanal = sonido[:, canal]
    else:
        uncanal = sonido

    # rango de observación en segundos
    inicia = 1.000
    termina = 2.002
    # observación en número de muestra
    a = int(inicia * muestreo)
    b = int(termina * muestreo)
    parte = uncanal[a:b]

    # Salida # Archivo de audio.wav
    print('archivo de parte[] grabado...')
    waves.write('parte01.wav', muestreo, parte)

    plt.plot(parte)
    plt.savefig("static/image/image12.png", depi=100)
    g2 = 'static/image/image2.png'
    return g2