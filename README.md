# DRAW - Detecting and Recognizing a Wild Range of cards

---

Ce détecteur utilise l'apprentissage profond pour détecter et classifier une carte Yu-Gi6oh! parmis les 11k+ existantes. 

Ce projet fait suite à celui de SuperZouloux (lien de la vidéo)

L'idée est de pouvoir remplacer les puces dans les sleeves qui sont assez contraignante et pouvoir ainsi reconnaire 
et localiser chaque carte 

---
## Overview de la méthode

Il est assez inanvisageable d'entrainer un detecteur directement sur l'ensemble des 11k+ cartes.
Ainsi il convient de séparer la tâche en plusieurs sous tâches. 

Dans un premier temps il s'agit de localiser la carte c'est ce que nous appèlerons la regression. Pour cela un des 
framworks les plus populaires et les plus performant à l'heure actuelle est le Yolo dans sa version la plus récente c'est-à-dire la version 8
développer par ultralytics.
Ainsi un jeu de donnée à été former et annoté à la main pour permettre l'entrainement du modèle

La deuxième tâche est ensuite la classification. Pour cela un classifier donnant de très bon résultat sur de grosse base de données à été
privilégié. Le Beit à été utiliser. Le transfert learning a permis de l'entrainer sur une plus petite base de donéne

---

