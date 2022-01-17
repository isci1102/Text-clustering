#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 20:55:23 2020

@author: huixin
"""

import pandas as pds
import copy
import spacy
import random
from math import*
import numpy as np
from numpy import*

#packages utilisés pour les nuages de mots
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import requests
import os

#on importe le package Spacy qui va nous servir pour la lemmatisation
nlp=spacy.load("fr_core_news_sm") 

debat_fiscalite = pds.read_csv('grand_debat_fiscalite.csv',delimiter = '#',dtype={"author_id": str, "id": object,"question": str,"libre": str,"reponse": str, }) #importer le dataframe
debat_fiscalite = debat_fiscalite[debat_fiscalite["libre"]== "True"]

debat_democratie = pds.read_csv('grand_debat_democratie.csv',delimiter = '#',dtype={"author_id": str, "id": object,"question": str,"libre": str,"reponse": str })
debat_democratie = debat_democratie[debat_democratie["libre"]== "True"]

debat_ecologie = pds.read_csv('grand_debat_ecologie.csv',delimiter = '#',dtype={"author_id": str, "id": object,"question": str,"libre": str,"reponse": str })
debat_ecologie = debat_ecologie[debat_ecologie["libre"]== "True"]

debat_service = pds.read_csv('grand_debat_services.csv',delimiter = '#',dtype={"author_id": str, "id": object,"question": str,"libre": str,"reponse": str })
debat_service = debat_service[debat_service["libre"]=="True"]


####Part 1: Nettoyage du texte
# Dans ces dataframes, on sélectionne la colonne sur laquelle on va travailler : les réponses libres
# Dans cette étape, on s'occupe aussi d'enlever la ponctuation et (renvoie une liste des textes sans ponctuation)

dfisc = list(debat_fiscalite['reponse'].str.strip().str.split('[\W_]+').str.join(' '))
reponsefisc = [str(texte) for texte in dfisc]

ddemo = list(debat_democratie['reponse'].str.strip().str.split('[\W_]+').str.join(' '))
reponsedemo = [str(texte) for texte in ddemo]

deco = list(debat_ecologie['reponse'].str.strip().str.split('[\W_]+').str.join(' '))
reponseeco = [str(texte) for texte in deco]

dser = list(debat_service['reponse'].str.strip().str.split('[\W_]+').str.join(' '))
reponseser = [str(texte) for texte in dser]

# On importe aussi la liste de mots vides (qui n’apportent pas de valeur informative) qui nous servira dans la partie de traitement du texte.
# Un mot vide est un mot communément utilisé dans une langue, non porteur de sens dans un document.
# Formellement, sa fréquence d’apparition est la même dans tous les documents. 
# De fait, les mots vides ne permettent pas de discriminer les documents

liste_mots_vides = pds.read_csv("mots_vides.txt")
MOTS_VIDES=list(liste_mots_vides['à']) #on les mets sous forme de liste
MOTS_VIDES.append('à') 

def lemmatisation(corpus):
    
    copy_corpus = copy.deepcopy(corpus) # on crée une copie du corpus
    #a l'aide du package spaCy, on crée une liste (corpus) de listes (textes) contenant les mots et leurs propriétés 
    return [[[mot.text,mot.lemma_, mot.pos_] for mot in nlp(texte)] for texte in copy_corpus] 
    # (mot du texte, format nomalisé, type)

print(lemmatisation(dfisc[:100])[:100])

#Fonction qui supprime les mots qui n'apportent pas de sens
def supprimer_mots (liste_corpus, mot_vide): 
    corpus = copy.deepcopy(liste_corpus)
    res = []
    tmp = []
    element_a_supprime = ['SPACE','DET','ADP','NUM','CCONJ','SCONJ','ADV','PRON', 'AUX', 'SCONJ', 'PROPN'] # liste des éléments à supprimer
    
    for texte in corpus:
        for mot in texte:
            if mot[2] not in element_a_supprime :
                if (mot[0] not in mot_vide) and (mot[1] not in mot_vide):
                    tmp.append(mot)
                    
        res.append(tmp)
        tmp = []
        
        
    return res
#print(supprimer_mots(lemmatisation(reponsefisc[:100]), MOTS_VIDES))

#On applique la lemmatisation aux quatres corpus (on teste pour un echantillon de 1000 textes)
LEMA_FISC = lemmatisation(reponsefisc[:1000])
LEMA_DEMO = lemmatisation(random.sample(reponsedemo,1000))
LEMA_ECO = lemmatisation(random.sample(reponseeco,1000))
LEMA_SER = lemmatisation(random.sample(reponseser,1000))

#On supprime les mots vides des corpus
FISC_NETOYE = supprimer_mots(LEMA_FISC, MOTS_VIDES) # on prend une liste de 500 réponses
DEMO_NETOYE = supprimer_mots(LEMA_DEMO, MOTS_VIDES)
ECO_NETOYE = supprimer_mots(LEMA_ECO, MOTS_VIDES)
SER_NETOYE = supprimer_mots(LEMA_SER, MOTS_VIDES)

#Lire et récupérer l’ensemble des mots d’un texte
#la fonction ci-dessous renvoie le corpus nettoyé sous forme d'une matrice où les lignes sont les textes et les colonnes des mots

def lire_mots(corpus_nettoyé):
    corpus = copy.deepcopy(corpus_nettoyé) #on garde une copie du corpus d'origine
    res =  [[mot[1] for mot in texte] for texte in corpus]
  #les mots selectionnées pour la suite de l'ananlyse sont la forme normalisée des mots de base

    return res
    
#print(lire_mots(FISC_NETOYE))
#On forme les matrices de textes pour chaque corpus
CORPUS_FISC = lire_mots(FISC_NETOYE)
CORPUS_DEMO = lire_mots(DEMO_NETOYE)
CORPUS_ECO = lire_mots(ECO_NETOYE)
CORPUS_SER = lire_mots(SER_NETOYE)


#Partie 2 : Analyse de texte

#Calculer la fréquence d’apparition d’un mot dans un texte
#La fonction suivante sert à calculer la fréquence d’apparition d’un mot dans un texte
def fréquence_mot(texte,mot):
    #on compte le nombre de fois que le mot apparait dans le texte et on divise par le nombre de mots dans ce texte
    res = texte.count(mot) / len(texte) 
    return res

# Calculer et stocker pour chaque texte les mots et leur fréquence. Stocker toutes ces informations pour un ensemble de textes disponibles dans un répertoire
#La fonction suivante calcule pour chaque texte d'un corpus la fréquence d'apparition d'un mot et les stocke dans un dictionnaire
def fréquence(corpus_nettoye):
    
    corpus = copy.deepcopy(corpus_nettoye) #on copie le corpus d'origine
    res = []
    tmp = {}
    
    # on calcule la frequence des mot de chaque texte et on stock dans un dictionnaire
    
    for texte in corpus:  #pour chaque texte
        for mot in texte: #pour chaque mot
            tmp[mot] = fréquence_mot(texte,mot) #on associe au mot la frequence d'apparition du mot dans le texte
            
     # on stocke les dictionnaires dans une liste où chaque case correspond à un texte  
    
        res.append(tmp)
        tmp = {}
    # on retourne une liste de dictionnaire 
    return res

#print(fréquence(CORPUS_FISC[:2]))
    
#On crée les variables globales correspondant au TF des mots de chaque texte dans chaque corpus
TF_FISC = fréquence(CORPUS_FISC) # variable globale correspondant au TF des mots 
                                # de chaque texte dans le copus de la fiscalité
                            
TF_DEMO = fréquence(CORPUS_DEMO) # variable globale correspondant au TF des mots 
                                 # de chaque texte dans le copus de la démocratie

TF_ECO = fréquence(CORPUS_ECO)  # variable globale correspondant au TF des mots 
                                # de chaque texte dans le copus de l'écologie

TF_SER = fréquence(CORPUS_SER) # variable globale correspondant au TF des mots 
                               # de chaque texte dans le copus des services
                               
#Calculer la fréquence d’apparition d’un mot dans un ensemble de texte (i.e. le pourcentage de fois où le mot apparaît dans un texte du corpus).
#la fonctions ci-dessous calcule l'idf de tous les mots d'un corpus mis en paramètre et renvoie un dictionnaire qui a pour cle le mot et pour valeur l'idf du mot
def calcul_IDF(corpus):
    
    taille_corpus = len(corpus) # nombre de textes dans le corpus
    liste_mot = []
    dic = {}
    resultat = {}
    nb_texte= 0
    idf_par_texte = 0
    
    
    for texte in corpus:
        for mot in texte:
            if mot not in liste_mot:
                liste_mot.append(mot)     # on recupère l'ensemble des mot du corpus

    
    for mot in liste_mot:
        for texte in corpus: # on commence par compter le nombre de textes qui contient le mot.
            if mot in texte:
                nb_texte +=1
        
        if nb_texte >0:
            idf_par_texte = log( taille_corpus / nb_texte)  # on calcule l'idf d'un mot en prend le log du rapport  
                                                 # de la taille du corpus et le nombre de fois où le mot apparait le textes.
        else:
            idf_par_texte = 0
            
        dic[mot] = idf_par_texte
        idf_par_texte = 0
        nb_texte = 0
        
    # le code ci-dessus nous calcule les idf des mots dans le corpus, mais pour plus discriminer les mots, nous allons favoriser
    # les mots qui revient plus souvent en donnant des scores 
        
    for texte in corpus:
        for mot in texte:
            if mot not in resultat:
                resultat[mot] = dic[mot]
            else:
                resultat[mot]+=dic[mot]
    
    return resultat
#print(calcul_IDF(CORPUS_FISC)) 

IDF_FISC = calcul_IDF(CORPUS_FISC) # variable globale correspondant à l'idf des mots 
                                  # de chaque texte dans le copus de la fiscalité
                                  
IDF_DEMO = calcul_IDF(CORPUS_DEMO) # variable globale correspondant à l'idf des mots 
                                  # de chaque texte dans le copus de la démocratie                           
                                  
IDF_ECO = calcul_IDF(CORPUS_ECO) # variable globale correspondant à l'idf des mots 
                                # de chaque texte dans le copus de l'écologie

IDF_SER = calcul_IDF(CORPUS_SER) # variable globale correspondant à l'idf des mots 
                                # de chaque texte dans le copus des services

#Calculer l’indice TF-IDF d’un mot d’un texte par rapport au corpus
#la fonction ci-dessous renvoie une liste de dictionnaire, chaque dictionnaire corepond à un texte, qui a pour cle un mot et pour valeur le tf-idf du mot .
def calcul_tfidf(tf,idf):
    
    liste_mot = [cle for cle in idf.keys()] # liste contenant les clés du dictionnaire
    res = []
    dic = {}
    tmp = 0
    
    # on commence par calculer le tfidf des mots d'un textes qu'on va stocker dans une dictionnaire
    # qui aura pour clé le mot et pour valeur son tfidf. On va stocker ce dictionnaire dans une liste
    # on fera ce processus pour tous les textes dans le corpus
    
    for texte in tf: #pour chaque texte
        for mot in list(texte.keys()): #pour chaque mot présent dans le dictionnaire des tf
            if mot in liste_mot:     #si ce mot apparait dans celui des idf
                tmp = idf[mot]*texte[mot] #alors on multiplie : TFxIDF
                dic[mot] = tmp         #et on stocke ce résultat dans le dictionnaire
        
        res.append(dic)  #on rassemble les dictionnaires dans une liste
        dic = {}
        tmp = 0
    
    return res
#print(calcul_tfidf(TF_FISC,IDF_FISC))                                  
                                  
TFIDF_FISC = calcul_tfidf(TF_FISC,IDF_FISC)# variable globale correspondant au tfidf des mots 
                                             # de chaque texte dans le copus de la fiscalité
TFIDF_DEMO = calcul_tfidf(TF_DEMO,IDF_DEMO) # variable globale correspondant au tfidf des mots 
                                             # de chaque texte dans le copus de la démocratie
TFIDF_ECO = calcul_tfidf(TF_ECO,IDF_ECO) # variable globale correspondant au tfidf des mots 
                                          # de chaque texte dans le copus de l'écologie
TFIDF_SER = calcul_tfidf(TF_SER,IDF_SER) # variable globale correspondant au tfidf des mots 
                                          # de chaque texte dans le copus des services


#Partie 3 : Nuage de mots

#Faire une fonction pour visualiser les mots importants d’un texte ou d’un ensemble de texte avec une représentation type nuage de mots (word cloud).
#la fonction nuage de mots prend un texte et pas une liste, ici on remet les mots importants en mode texte à partie de la matrice
def retexte (corpus):
    res=" "
    for texte in corpus:
        for mot in texte:
            res=res+" "+mot
    return res
#print (retexte(CORPUS_FISC))
                
TEXTE_FISC=retexte(CORPUS_FISC)
#TEXTE_SER=retexte(CORPUS_SER)
#TEXTE_ECO=retexte(CORPUS_ECO)
#TEXTE_DEM=retexte(CORPUS_DEMO)                                          

#on initialise les paramètres du nuage de mots
def wordclouds(corpus):
    #voici les paramètre par défaut d'un wordcloud
    width = int (4000)
    height = int (2000)
    prefer_horizontal = float (0.10)
    mask = np.array (None)
    contour_width = float (0)
    scale = float (0.03)
    min_font_size = int (4)
    font_step = int (1)
    max_words =100
    background_color ="white"
    mode = "RGB"
    relative_scaling : float (.5)
    color_func=lambda *args, **kwargs: "green"

    wordcloud = WordCloud(max_font_size=100, max_words=100,prefer_horizontal=0.9,background_color ="white"
    ,contour_width=0.5,relative_scaling=1,mode="RGB").generate(corpus)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    
wordclouds(TEXTE_FISC)

#Voici une autre manière plus personnalisée d'un nuage de mot, on a rajouté un masque et personnalisé les couleurs.
def wordclouds2(corpus):
    
    france_mask = array(Image.open("FRANCE4.jpg")) #on créer le masque 

    
    width = int (4000)
    height = int (2000)
    prefer_horizontal = float (0.10)
    mask = array (None)
    contour_width = float (0)
    scale = float (0.03)
    min_font_size = int (4)
    font_step = int (1)
    max_words =300
    background_color ="white"
    mode = "RGBA"
    relative_scaling : float (.5)
    color_func=lambda *args, **kwargs: "green"
    colormap="seismic"

    wordcloud = WordCloud (colormap="seismic",margin=1, min_font_size = int (9), max_font_size=200, max_words=300,prefer_horizontal=0.9,background_color ="white"
    ,contour_width=0.5,relative_scaling=1,mask=france_mask, contour_color='white',random_state=1).generate(corpus)
    
    plt.figure(figsize=(20,10),facecolor = 'white')
    plt.tight_layout(pad=0)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()



#Partie 4 : Distance entre texte

#Déterminer l’ensemble des n mots les plus pertinents dans le corpus ( n est un paramètre, la notion de pertinence est basée sur IDF)
def motspluspertinents(idf_corpus,n):
    
    res = []
    tmp = sorted(idf_corpus.items(), key = lambda x : x[1],reverse = True )[:n] # trie par ordre decroissant le dictionnaire des idf
    res=[mot[0] for mot in tmp] #on ajoute les mots dans une liste
        
    return res
    
#print(motspluspertinents(IDF_FISC,100))
    
MOTS_PERTINENTS_FISC = motspluspertinents(IDF_FISC,100)

#nous avons crée une fonction qui trace le diagramme en barre horizontal des mots pertinents de manière à voir lesquels ont le plus grand idf et les différences d'idf entre ces mots.
def plot_mot_pertinents(idf_corpus,n):
    
    mot_pertinent = motspluspertinents(idf_corpus,n)
    res = {mot: idf_corpus[mot] for mot in mot_pertinent}
    tmp = sorted(res.items(), key = lambda x : x[1]) # on trie pour avoir le graphique dans l'ordre décroissant
    mots = [mot[0] for mot in tmp]
    idf = [mot[1] for mot in tmp]
    
    plt.figure(figsize=(10,8), dpi=100)
    plt.barh(mots, idf, color=[ 'darkblue','mediumblue', 'royalblue','blue', 'cornflowerblue', 'lightsteelblue'])
    plt.yticks(mots, mots, rotation = 'horizontal')
    plt.title("Exemple de mots pertinents selectionnés", color = 'blue')
    plt.xlabel("Idf")
    plt.ylabel("Les mots Pertinents")
    plt.show()
    
plot_mot_pertinents(IDF_FISC,40)

# Transformer un texte en un vecteur de fréquences sur les n  mots les plus pertinents 
#Nous allons faire une fonction qui prend un corpus et la liste de mots pertinents en parametre et renvoit un tableau de tfidf avec les mots pertinents en colonnes et les textes en ligne:

def vectorizer_TFIDF(corpus,mot_pertinent):
    
    tf = fréquence(corpus) #on calcule le tf des mots
    idf = calcul_IDF(corpus)  #, leur idf
    tfidf_corpus = calcul_tfidf(tf, idf) # et leur tf idf
    tmp = []
    vecteur = {}
    
    for tfidf_texte_index in range(len(tfidf_corpus)):  #pour chaque texte
        for mot in mot_pertinent: #pour chauqe mot pertinent
            if mot not in tfidf_corpus[tfidf_texte_index].keys(): 
                tmp.append(0)
            else:
                tmp.append(tfidf_corpus[tfidf_texte_index][mot]) #on ajoute leur tf idf a la liste
         
        if tmp != list(np.zeros(len(tmp))): # on veut des textes qui comportent au moins un mot pertinent
            vecteur[tfidf_texte_index]=tmp 
        tmp = []
        
    #resultat = np.array(vecteur)
    resultat = pds.DataFrame(data = vecteur, index = mot_pertinent)
    resultat = resultat.transpose()
                
    return resultat

VECTFISC=vectorizer_TFIDF(CORPUS_FISC, MOTS_PERTINENTS_FISC)

#Calculer la distance cosinus entre deux textes représentés par leur vecteur de fréquence respectif
#On calcule la similarité cosinus entre deux textes représentés par leur vecteur de TF IDF respectif
def distance_cosinus(vecteur_texte1, vecteur_texte2):
    
    texte1 = list(vecteur_texte1)
    texte2 = list(vecteur_texte2)
    res = 0
    scalaire = 0
    norme_texte1 = 0
    norme_texte2 = 0

    if len(texte1)==len(texte2):
        for i in range(len(texte1)):
            scalaire += texte1[i]*texte2[i] # étape du calcule de scalaire texte1 et texte2
            norme_texte1 += texte1[i]**2
            norme_texte2 += texte2[i]**2
        res = scalaire / (sqrt(norme_texte1) * sqrt(norme_texte2))
    else:
        print('Les dimensions ne sont pas égales, revérifiez les!')
        
    return res
print(distance_cosinus(VECTFISC.loc[0],VECTFISC.loc[1]))

# On a crée cette fonction pour verifier si notre similarité cosinus etait efficace: le but est de choisir un texte et d'obtenir celui qui lui ressemble le plus dans le même corpus
def maxsimilaire(vect_corpus,texte_index):
    
    res = {}
    if texte_index in vect_corpus.index: 
        for numtexte in range(vect_corpus.shape[0]): #on parcours les textes
            if numtexte in vect_corpus.index:
                res[numtexte] = distance_cosinus(vect_corpus.loc[texte_index], vect_corpus.loc[numtexte])  
                #on sauvegarde les similarités cosinus

        tmp = sorted(res.items(), key = lambda x : x[1],reverse = True )[:2] # trie par ordre decroissant

        if tmp[0][0] != texte_index:
            return tmp[0] #on retourne le max mais pas le premier car il correspond au texte lui même
        else:
            return tmp[1]
        
    else:
        print("Ce texte n'est pas dans le dataFrame")
    
    
#print(maxsimilaire(VECTFISC,6))
#print(VECTFISC.loc[6],VECTFISC.loc[361])                                           
                                  

##Partie 5 : Classification non supervisée                                  
def kmeans(data_tfidf, nbcluster,nbrepeter): 
    sommedistance={} #créer un dictionnaire pour conserver les sommes des distance 
    for i in range(nbrepeter): #on répère pour trouver un qui donne le plus grand la somme des distances cosinus
     
        data_copy = copy.deepcopy(data_tfidf)
        
        #le min et le max de tfidf de chaque texte
        min_max=[[min(data_tfidf.loc[i]),max(data_tfidf.loc[i])]  for i in data_tfidf.index] 
        
        # on va definir nbcluster centroids aléatoire, c'est l'initialisation des centroids
        # on appelle centroids les centres des clusters
        
        centroids={
                j+1: [np.random.random()*(min_max[i][1]-min_max[i][0]) for i in range(len(data_tfidf.columns))] 
                for j in range(nbcluster)}
        
        # initialisation de notre programme
        
        # on va calculer et stocker la distance entre chaque texte et les centroids
        for i in centroids.keys():
            data_copy['distance_du_centroids_{}'.format(i)] = np.asarray([distance_cosinus(centroids[i], data_tfidf.loc[index_texte]) 
                                                        for index_texte in data_tfidf.index])
                 
    
        centroid_distance_cols = ['distance_du_centroids_{}'.format(i) for i in centroids.keys()] # les labels
        data_copy['le_plus_proche'] = data_copy.loc[:, centroid_distance_cols].idxmax(axis=1) 
        # data_tfidf['le_plus_proche'] donne pour un texte, son centroids le plus proche
        data_copy['le_plus_proche'] = data_copy['le_plus_proche'].map(lambda x: int(x.lstrip('distance_du_centroids_')))
        # enlève 'distance_du_centroids_'
          
        while True: 
            
            proche_centroids = data_copy['le_plus_proche'].copy(deep=True)
            
            # déplacer les centres à la moyenne de leurs membres
            for i in centroids.keys():
                for j in range(len(data_tfidf.columns)):
                    centroids[i][j] = np.mean(data_copy[data_copy['le_plus_proche'] == i][data_tfidf.columns[j]])
        
            for i in centroids.keys():
                data_copy['distance_du_centroids_{}'.format(i)] = np.asarray([distance_cosinus(centroids[i], data_tfidf.loc[index_texte]) 
                                                        for index_texte in data_tfidf.index])
    
            centroid_distance_cols = ['distance_du_centroids_{}'.format(i) for i in centroids.keys()] # les labels
            
            data_copy['le_plus_proche'] = data_copy.loc[:, centroid_distance_cols].idxmax(axis=1)
            # data_tfidf['le_plus_proche'] donne pour un texte, son centroids le plus proche
            
            data_copy['le_plus_proche'] = data_copy['le_plus_proche'].map(lambda x: int(x.lstrip('distance_du_centroids_')))
            # enlève 'distance_du_centroids_'
    
           
            # fini, si on a le même résultat comme étape précédente i.e si  proche_centroids==data_tfidf['le_plus_proche'])
            if proche_centroids.equals(data_copy['le_plus_proche']): 
                break 
        clusters = {
            'cluster_{}'.format(i): data_copy[data_copy['le_plus_proche']==i].index # on veut récuperer l'index des textes dans le clusters
            for i in centroids.keys()}
        
        #conserve la somme des distance 
        ssd = 0
        for k in range(1,nbcluster+1):
            cluster_k = data_copy[data_copy['le_plus_proche'] == k]
            ssd=ssd+cluster_k['distance_du_centroids_{}'.format(k)].sum()
         
        if ssd not in sommedistance:
            sommedistance[ssd]=[data_copy,clusters,centroids]   
    best=max(list(sommedistance.keys()))   #on trouve le data qui a la sommes de distance la plus grand
    
    
    return sommedistance[best]

#trouver le meilleur nombre de cluster 
def methodecoude(data_tfidf,max_clusters):
    SSEtotal = []
    
    for i in range(1,max_clusters+1):
        data_tfidf0 = copy.deepcopy(data_tfidf)
        data_tfidf0 = kmeans(data_tfidf0,i,50)[0]
        sse = 0
        for k in range(1,i+1):
            cluster_k = data_tfidf0[data_tfidf0['le_plus_proche'] == k]
            sse=sse+cluster_k['distance_du_centroids_{}'.format(k)].sum()

        SSEtotal.append(sse) 
    print(SSEtotal)
    plt.plot(range(1,max_clusters+1),SSEtotal)
    plt.title('Méthode de coude')
    plt.xlabel('nombre de clusters')
    plt.ylabel('somme des distances')
    plt.show()



##LES RESULTAT                

#FISCAL
methodecoude(VECTFISC,5)
CLUSTER_FISC = kmeans(VECTFISC,4,50)[1]
for cluster in CLUSTER_FISC.keys():
    wordclouds2(retexte([CORPUS_FISC[i] for i in CLUSTER_FISC[cluster]]))
                                  
#définir une fonction pour obtenir directement les vecteurs, n c'est le nombre de mots pertinent qu'on veut obtenir
def getvector(corpus,n):
    IDF = calcul_IDF(corpus)
    MOTS_PERTINENTS = motspluspertinents(IDF,n)
    VECT=vectorizer_TFIDF(corpus, MOTS_PERTINENTS)
    return VECT                                  

VECTDEMO=getvector(CORPUS_DEMO,100)
VECTECO=getvector(CORPUS_ECO,100)
VECTSER=getvector(CORPUS_SER,100)


#DEMO
methodecoude(VECTDEMO,5)
CLUSTER_DEMO = kmeans(VECTDEMO,3,50)[1]
for cluster in CLUSTER_DEMO.keys():
    wordclouds2(retexte([CORPUS_DEMO[i] for i in CLUSTER_DEMO[cluster]]))

#ECO
methodecoude(VECTECO,5)
CLUSTER_ECO = kmeans(VECTECO,4,50)[1]
for cluster in CLUSTER_ECO.keys():
    wordclouds2(retexte([CORPUS_ECO[i] for i in CLUSTER_ECO[cluster]]))

#SER
methodecoude(VECTSER,5)
CLUSTER_SER = kmeans(VECTSER,3,50)[1]
for cluster in CLUSTER_SER.keys():
    wordclouds2(retexte([CORPUS_SER[i] for i in CLUSTER_SER[cluster]]))

#TOTAL

#définir une fonction pour fusionner les corpus des 4 thèmes
def merge(x,y,z,q):
    texttotal=[x,y,z,q]
    final_list=[]
    for liste in texttotal:
        for ligne in liste:
            final_list.append(ligne)
    return final_list    

CORPUS_TOTAL=merge(CORPUS_FISC,CORPUS_DEMO,CORPUS_ECO,CORPUS_SER)
VECTTOTAL=getvector(CORPUS_TOTAL,400)

CLUSTER_TOTAL = kmeans(VECTTOTAL,4,50)[1]
for cluster in CLUSTER_TOTAL.keys():
    wordclouds2(retexte([CORPUS_TOTAL[i] for i in CLUSTER_TOTAL[cluster]]))





