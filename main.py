from flask import Flask, escape, request,render_template
import scipy.io.wavfile as waves
app = Flask(__name__)
import  re
import os
#os.chdir('..')
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
# Load the LDA model from sk-learn
from sklearn.decomposition import LatentDirichletAllocation as LDA

import math
import numpy as np
from bs4 import BeautifulSoup
from urllib.request import urlopen
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from collections import Counter
from scipy import spatial
import io
from pydub import AudioSegment
from pydub.utils import make_chunks
# Imports the Google Cloud client library
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
import os
import io, re, nltk, collections, os, numpy as np, pandas as pd, math, seaborn as sns, subprocess, shlex


nltk.download('stopwords')

def limpieza( d):
    n1 = re.sub('[^a-zA-Z0-9 \n\.]', '', d)
    n1 = re.sub('[.]', '', n1)
    n1 = re.sub('[0-9]', '', n1)
    n2 = n1.lower()
    #n3 = n2.split()
    return n2
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

    tit = ''
    for links in soup.find_all('pre'):
        tit=links.text
    print('=======IMPRESION DE LOS TITULOS=======')
    print(tit)
    return tit

def jaccard_similarity(list1, list2):
    s1 = set(list1)

    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))

def toquenizar(DATA): #recibe como parametro un diccionario de documentos y regrea un bagofword en matriz
    dd = ''
    for j in range((len(DATA))):
        g = list(DATA[j].values())
        h = g[0]
        k = re.sub('[?|$|.|!|(|)|,]', r' ', h.lower())
        dd = dd +" "+k
    d1 = dd.split()
    d=stopWStem(d1)#METODO
    dic = dict(Counter(d))
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
        r = []
        for b in range((len(DATA))):
            g = list(DATA[b].values())
            h = g[0]
            k = re.sub('[|"|?|$|.|!|(|)|,]', r' ', h.lower())
            d2 = k.split()
            d=stopWStem(d2)#METODO
            for j in range(len(d)):
                if dic[i] == d[j]:
                    n = n + 1
                    s.append(j + 1)  # guardanso la posicion
            matrix[i][b] = n
            if n != 0 and r != None:
                r.append(b + 1)
                r.append(n)  # numero de repitencia
                r.append(s)
                y.append(r)
                res = {dic[i]: y}
                resultado.update(res)
            s = []
            n = 0
            res = {}
            r = []
        y = []
    print('RESULTADO FINAL FULL INVERTED INDEX')
    print(resultado)
    print(matrix)
    return matrix


def pesoTF(ft,d): #recibe como paramnetro una bolsaa de palabras tf , d=longitud del vector
    print('PESADO DEL TF ')
    pesadotf1 = np.empty((len(ft), len(d)))
    for i in range(len(ft)):
        for j in range(len(d)):
            if ft[i][j] > 0:
                tf =round( 1 + (math.log(ft[i][j], 10)),2)

                pesadotf1[i][j] = tf
            else:
                pesadotf1[i][j] = 0
    print(pesadotf1)
    print(len(pesadotf1))
    return pesadotf1


def NormalizarMatriz(pesadotf1,d): #recibe como parametro el pesado del tf y retorna un matriz ya normalizada d=logitud del vector
    print('NORMALIZACION DE LA MATRIZ')
    matrixNorma = np.empty((len(pesadotf1), len(d)))
    for j in range((len(d))):
        mod = np.sqrt((pesadotf1[:, j] ** 2).sum())
        for b in range((len(pesadotf1))):
            c = round((pesadotf1[b][j] / mod), 2)
            matrixNorma[b][j] = c
    print(matrixNorma)
    return matrixNorma

def matrizSimilitud(matrixNorma,d ):#recibe una matriz normalizada d=logitud del vector
    matrixSimi = np.empty((len(d), len(d)))
    for i in range(len(d)):
        vec1 = matrixNorma[:, i]
        for j in range(len(d)):
            vec2 = matrixNorma[:, j]
            result = round((1 - spatial.distance.cosine(vec1, vec2)), 2)
            matrixSimi[i][j] = result
    print('Matriz de similitud')
    print(matrixSimi)
    return matrixSimi


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/USER/PycharmProjects/Speech_Reco/tony.json"


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
    command = "ffmpeg -i " + audio_file_name + " " + audio_file_name.split('.')[0] + ".wav"
    command = str(command)
    print(command)
    subprocess.Popen(shlex.split(command))
    wav_file = audio_file_name.split('.')[0] + '.wav'
    print(wav_file)
    return wav_file


def stereo_to_mono(audio_file_name):
    sound = AudioSegment.from_wav(audio_file_name)
    sound = sound.set_channels(1)
    sound.export(audio_file_name, format="wav")

def final(file):
    fh = open("recognized.txt", "w+")
    recon=''
    #mp3_to_wav(file)

    stereo_to_mono(file)

    myaudio = AudioSegment.from_file(file, "wav")
    chunk_length_ms = 50000  # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms)  # Make chunks of one sec

    for i, chunk in enumerate(chunks):
        chunk_name = "chunk{0}.wav".format(i)
        print("exporting", chunk_name)
        chunk.export(chunk_name, format="wav")

        recon= recon+' '+ transcribe_file(chunk_name)

        fh.write(recon + ". ")
    fh.close()
    return  recon


#man=np.array([[0.789,0.832,0.524],[0.515,0.555,0.465],[0.335,0,0.405],[0,0,0.588]])
#l=[1,2,3]
#matrizSimilitud(man,l)

def plot_10_most_common_words(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts += t.toarray()[0]

    count_dict = (zip(words, total_counts))

    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words))

    plt.figure(2, figsize=(15, 15 / 1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90)
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.savefig("static/image/image1.png", depi=100)
    plt.savefig
    g='static/image/image1.png'
   # plt.show()
    return g


def print_topics(model, count_vectorizer, n_top_words):
    top = ''
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        top=top+','+(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print(top)
    return top

# Tweak the two parameters below
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
    plt.savefig("static/image/image2.png", depi=100)
    g2 = 'static/image/image2.png'
    return g2




@app.route('/')
def fi()->'html':

    return render_template('home.html')



@app.route('/cogerpath',methods=['POST'])
def cogerpath()->'html':

    #nombreUser = request.form['nombreUser']
    #path = request.args.get('myfile')
    path = request.form['myfile']
   #h=final(path)
    print(path)
    if((path.split('.')[1])=='mp3'):
        print(path.split('.')[1])
        path1=mp3_to_wav(path)

    else:
        path1=path
        print('estoy dentro del else',path1)
    fh2 = open("original.txt", "w+")
    musica = scraping(
        'https://www.lyrics.com/lyric/33554233/La+La+Land+%5BOriginal+Motion+Picture+Soundtrack%5D/City+of+Stars')
    fh2.write(musica + ". ")
    fh2.close()
    wb = limpieza(musica)
    musica1 = musica.split('\n')
    wb = wb.split()
    wb1 = stopWStem(wb)
    print(wb1)
    spe = final(path1)
    spe1 = spe.split('.')
    sp = limpieza(spe)
    sp = sp.split()
    sp1 = stopWStem(sp)
    print(sp1)
    jac=jaccard_similarity(wb1, sp1)
    jac1=(round(jac,2))*100
    print(jac)

    #print(jaccard_similarity(wb1, sp1))

    datos = [{'letra': musica}, {'speech': spe}]

    print(datos)
    r1 = toquenizar(datos)
    r2 = pesoTF(r1, datos)
    r3 = NormalizarMatriz(r2, datos)
    r4 = matrizSimilitud(r3, datos)
    cos=(round(r4[0][1],2))*100
    if (cos>90):
        fhh = open("C:/Users/USER/PycharmProjects/Speech_Reco/recognized.txt", "r");

    else:
        fhh = open("C:/Users/USER/PycharmProjects/Speech_Reco/original.txt", "r");



    papers = fhh.read()
    fhh.close()
    papers = re.sub('[,\.!?]', '', papers)
    papers = papers.lower()
    papers = papers.split()

    print(papers)
    long_string = ','.join(list(papers))
    print(long_string)
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
    wordcloud.generate(long_string)
    # Visualize the word cloud
    wordcloud.to_image()

    # Initialise the count vectorizer with the English stop words
    count_vectorizer = CountVectorizer(stop_words='english')
    # Fit and transform the processed titles
    count_data = count_vectorizer.fit_transform(papers)
    image= plot_10_most_common_words(count_data, count_vectorizer)
    image2=graphSpectr(path1)

    number_topics = 5
    number_words = 4
    # Create and fit the LDA model
    lda = LDA(n_components=number_topics)

    lda.fit(count_data)

    lda.fit(count_data)
    # Print the topics found by the LDA model
    print("Topics found via LDA:")
    d = print_topics(lda, count_vectorizer, number_words)
    print(d)
    d = d.split(',')[1]


    return render_template('home.html',results=musica,speech=spe1,jaccard=jac1,coseno=cos,title=path,recomend=d,im=image,im2=image2)

app.run()