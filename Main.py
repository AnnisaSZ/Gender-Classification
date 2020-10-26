import pandas as pd
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

#filtering
def filtering(data):
    regexURL = data.str.replace(r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})', ' ')
    regexUsername = regexURL.str.replace(r'\S*@\S*\s?', ' ')
    regexContainNumber = regexUsername.str.replace(r'\w*[0-9]\w*', ' ')
    regexNotAlphabet = regexContainNumber.str.replace('[^A-Za-z]+', ' ')
    return (regexNotAlphabet)
#case folding
def caseFolding(data):
    return data.str.lower()
#tokenisasi
def tokenisasi(data):
    return data.str.split()
#stemming
def stemming(data):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    documents=[]
    for i in range(len(data)):
        documents.append([stemmer.stem(word) for word in data[i]])
    return documents

#term unik
def termUnik(data):
    y=[]
    x=np.array(data)
    for i in range(len(x)):
        y+=x[i]
    term=np.unique(y)
    return (term)
#tf, df
def hitTF(data, termUnik):
    tf=[]
    for i in range (len(termUnik)):
        temp=[]
        for j in range (len(data)):
            jumlah=(data[j].count(termUnik[i]))
            temp.append(jumlah)    
        tf.append(temp)
    return (tf)

#hitung df
def hitDf(data):
    df=[]
    for i in range (len(data)):
        jumlah=0
        for j in range (len(data[i])):
            if (data[i][j]>0):
                jumlah+=1
        df.append(jumlah)
    return df

#lc, avlc
def hitLc(data):
    lc=[]
    temp=np.transpose(data)
    for i in range (len(temp)):
        lc.append(np.sum(temp[i]))
    avl=np.average(lc)
    return lc,avl
#Tf, df uji
def hitTFUji(data, termUnik, tfLatih):
    tf=[]
    df=[]
    for i in range (len(data)):
        temptf=[]
        tempdf=[]
        for j in range (len(termUnik)):
            temp=[]
            if(data[i].count(termUnik[j])!=0):
                temp=tfLatih[j]
            else:
                temp=np.zeros(len(tfLatih[0]))
            temptf.append(temp)
            tempdf.append(np.sum(temp))
        tf.append(temptf)
        df.append(tempdf)
    return (tf, df)

#Menghitung nilai idf
def hitIdf(data, ldok):
    idf = []
    for i in range(len(data)):
        temp=[]
        for j in range(len(data[0])):
            if (data[i] != 0):
                temp.append(np.log10((ldok-data[i][j]+0.5)/(data[i][j]+0.5)))
            else:
                temp.append(0)
        idf.append(temp)
    return idf

#BM25
def hitBM25(idf,termUji,lc,avlc):
    k1=1.2
    b=0.75
    bm25=[]
    for k in range(len(termUji)):
        temp2=[]
        for i in range(len(termUji[0][0])):
            temp=[]
            for j in range(len(termUji[0])):
                temp.append(idf[k][j]*((k1+1)*termUji[k][j][i])/(k1*((1-b)+b*(lc[i]/avlc))+termUji[k][j][i])) 
            temp2.append(np.sum(temp))
        bm25.append(temp2)
    return (bm25)
#klasifikasi
def KNN(BM25, label, k):
    final=[]
    for j in range(len(BM25)):
        kelas=[]
        data=np.array(BM25[j])
        idx=(-data).argsort()[:k]
        for i in idx:
            kelas.append(label[i])
        perempuan=kelas.count('p')
        lakilaki=kelas.count('l')
        if(perempuan>lakilaki):
            final.append('p')
        else:
            final.append('l')
    return(final)

'''
Data Latih
'''
#baca file
datalatih = pd.read_csv('sampleLatih.csv')
datauji=pd.read_csv('sampleUji.csv')

tweet=datalatih['tweet']
label=datalatih['JK']
k=3

#filtering
fltr=filtering(tweet)
#case folding
cf=caseFolding(fltr)
#tokenisasi
token=tokenisasi(cf)
#stemming
stem=stemming(token)

#term unik
term=termUnik(stem)

#nilai tf dan df
tfLatih=hitTF(stem,term)
df=hitDf(tfLatih)
#hitung nilai panjang data dan rata rata
lc, avlc=hitLc(tfLatih)
N=len(lc)

#Data Uji
#filtering
fltrUji=filtering(datauji['tweet'])
jksebenarnya=datauji['jk']
#case folding
cfUji=caseFolding(fltrUji)
#tokenisasi
tokenUji=tokenisasi(cfUji)
#stemming
stemUji=stemming(tokenUji)

#tf df data uji
tfUji,dfUji=hitTFUji(stemUji,term,tfLatih)
#idf uji
idf=hitIdf(dfUji,N)
#bm25
bm25=hitBM25(idf,tfUji,lc,avlc)
#KNN
klasifikasi=KNN(bm25, label, k)

print("Hasil klasifikasi data: ",klasifikasi)
