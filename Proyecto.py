# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 23:00:47 2019

@author: tony
"""
from bs4 import BeautifulSoup
from urllib.request import urlopen
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from collections import Counter

import speech_recognition as sr
import webbrowser as wb
import numpy as np

import io
from pydub import AudioSegment
from pydub.utils import make_chunks
# Imports the Google Cloud client library
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/USER/PycharmProjects/Speech_Reco/tony.json"

def stopWStem(d1):
    n4 = stopwords.words('english')

    for i in range(len(n4)):
        while (d1.count(n4[i])):
            d1.remove(n4[i])

    stemmer = PorterStemmer()
    n5 = stemmer.stem('solutions')
    d = []
    for st in d1:
        d.append(stemmer.stem(st))
    return d

def scraping( link):
    file = urlopen(link)
    html = file.read()
    file.close()
    soup = BeautifulSoup(html)  ## para indexar
    busca = soup.find_all("<span style=font-family: arial; font-size: small;>")  ## busca tods las etiquets "a"

    tit = []
    for links in soup.find_all('pre'):
       # print(links.get('href'))  ## busca dentro con subetiquetas


        tit.append(links.text)
    #titulos = tit[6:7]
    print('=======IMPRESION DE LOS TITULOS=======')
    print(tit)
    return tit


def toquenizar(DATA): #recibe como parametro un diccionario de documentos y regrea un bagofword en matriz
    dd = ''
    for j in range((len(DATA))):
        g = list(DATA[j].values())
        h = g[0]
        #k = re.sub('[?|$|.|!|(|)|,]', r' ', h.lower())
        dd = dd +" "+h
    d1 = dd.split()
    #d=stopWStem(d1)#METODO
    dic = dict(Counter(d1))
    dic = list(dic)
    n = []
    s = []
    l = 0
    n = 0
    r = []
    res = {}
    resultado = {}
    y = []
    print('tamaño del cidcionario',len(dic))
    print(dic)
    matrix=np.empty((len(dic), len(DATA)))
    for i in range(len(dic)):
        #print('La letra :', end="  ")
        #print(dic[i], end="  ")
        r = []

        for b in range((len(DATA))):
            g = list(DATA[b].values())
            h = g[0]
            #k = re.sub('[?|$|.|!|(|)|,]', r' ', h.lower())
            d2 = h.split()
            #d=stopWStem(d2)#METODO
            d=d2
            for j in range(len(d)):

                if dic[i] == d[j]:
                    n = n + 1
                    s.append(j + 1)  # guardanso la posicion
            # print('Documento', end="  ")
            # print(b)
            #print(n)
            matrix[i][b] = n
            if n != 0 and r != None:
                r.append(b + 1)
                r.append(n)  # numero de repitencia
                r.append(s)
                y.append(r)
                res = {dic[i]: y}
                resultado.update(res)
                #print(r, end="  ")
            s = []
            n = 0
            res = {}
            r = []
        #print(i)
        y = []
        #print()
        #print('========================================================================================')

    print('RESULTADO DE PALABRAS DEL DICCIONARIO')
    print(dic)
    print()
    print('RESULTADO FINAL FULL INVERTED INDEX')
    print(resultado)
    #return resultado
    print(matrix)
    return matrix
    # print(resultado.get('to'))

def transcribe_file(speech_file):
    """Transcribe the given audio file."""
    client = speech.SpeechClient()
    prediccion = " "
    with io.open(speech_file, 'rb') as audio_file:
        content = audio_file.read()

    audio = types.RecognitionAudio(content=content)
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code='en-US')

    response = client.recognize(config, audio)
    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    for result in response.results:
        # The first alternative is the most likely one for this portion.
        prediccion = prediccion + " " + result.alternatives[0].transcript
    return prediccion


def mp3_to_wav(audio_file_name):
    if audio_file_name.split('.')[1] == 'mp3':
        sound = AudioSegment.from_mp3(audio_file_name)
        audio_file_name = audio_file_name.split('.')[0] + '.wav'
        sound.export(audio_file_name, format="wav")


def stereo_to_mono(audio_file_name):
    sound = AudioSegment.from_wav(audio_file_name)
    sound = sound.set_channels(1)
    sound.export(audio_file_name, format="wav")


file = "city_of_stars.wav"
mp3_to_wav(file)
stereo_to_mono(file)

myaudio = AudioSegment.from_file(file, "wav")
chunk_length_ms = 50000  # pydub calculates in millisec
chunks = make_chunks(myaudio, chunk_length_ms)  # Make chunks of one sec

for i, chunk in enumerate(chunks):
    chunk_name = "chunk{0}.wav".format(i)
    print("exporting", chunk_name)
    chunk.export(chunk_name, format="wav")
    print(transcribe_file(chunk_name))




def speechRecognition(sound):
    with sr.AudioFile(sound) as source:
        #r1.adjust_for_ambient_noise(source)
        audio = r1.listen(source)

        try:
            text = r1.recognize_google(audio)
            print('Audio is: \n',text)
        except Exception as e:
            print('shet', e)
    return text

sound = "Prueba3.wav"
text1=speechRecognition(sound)

dc={}
DATA1 = []

dc['doc']=text1
DATA1.append(dc)
#toquenizar(DATA1)
         
titulo=scraping('https://www.lyrics.com/lyric/33554233/La+La+Land+%5BOriginal+Motion+Picture+Soundtrack%5D/City+of+Stars')
for i in range(len(titulo)):
    print(titulo[i])
    dc['doc']=titulo[i]
    DATA1.append(dc)
    dc={}

print('=======IMPRIMIENDO LA LISTA DE TITULOS=======')
print(DATA1)
print()
toquenizar(DATA1)
            
                