<div style="text-align: center;">
    <p>
        <img src="figures/banner-draw.png">
    </p>

[🇬🇧 English](README.md)

<div>

![Licence](https://img.shields.io/github/license/Ileriayo/markdown-badges?style=flat)
![Medium](https://img.shields.io/badge/Medium-12100E?style=flat&logo=medium&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)

</div>
DRAW est le tout premier détecteur d'objets entraîné à détecter les cartes Yu-Gi-Oh! dans tous types d'images, 
et en particulier dans les images de duels.

D'autres travaux existent (voir ...) mais aucun n'est capable de reconnaître des cartes pendant un duel.

DRAW est entièrement open source et toutes les contributions sont les bienvenues.
</div>

## <div style="text-align: center;">Documentation</div>

<details open>
<summary>Installation</summary>
</details>

<details open>
<summary>Usage</summary>
</details>

---
## <div style="text-align: center;">Modèles and Données</div>

<details open>
<summary>Modèles</summary>

Dans ce projet, les tâches ont été divisées de manière à ce qu'un modèle localise les cartes et qu'un autre les classifie. 
De même, pour classifier les cartes, j'ai divisé la tâche de manière à ce qu'il y ait un modèle pour chaque type de carte,
le modèle à utiliser étant déterminé par la couleur de la carte.

Les modèles sont disponible au téléchargement sur <a href="https://huggingface.co/HichTala/draw">Hugging Face</a>. 
Les modèles commençant par `beit` représentent la classification et celui commençant par `yolo` la localisation.

Pour l'instant, seuls les modèles pour le jeu "rétro" sont disponibles, mais ceux pour le format classique seront bientôt ajoutés.


J'ai considéré comme format "rétro" toutes les cartes antérieures au premier set _syncro_,
donc toutes les cartes éditées jusqu'au set Lumière de la Destruction (LODT - 13/05/2008) et toutes les cartes de speed duel.
</details>

<details open>
<summary>Données</summary>

Pour créer un jeu de données, l'api <a href="https://db.ygoprodeck.com/api-guide-v2/">YGOPRODeck</a> a été utilisée. 
Deux jeux de données ont ainsi été constitué, l'un pour le jeu "rétro" et l'autre pour le jeu au format classique.
De la même manière qu'il y a un modèle par type de cartes, il y a un jeu de données par type de cartes.

Les jeux de données sont disponible au téléchargement sur <a href="https://huggingface.co/datasets/HichTala/yugioh_dataset">Hugging Face</a>.

Pour l'instant, seul les jeux de données "rétro" sont disponible, mais ceux pour le format classique seront bientôt ajoutés.

</details>

---
## <div style="text-align: center;">Inspiration</div>

Ce projet a été inspiré par un projet du créateur [SuperZouloux](https://www.youtube.com/watch?v=64-LfbggqKI) 
donnant vie aux cartes _Yu-Gi-Oh!_ à l'aide d'un hologramme. Son projet utilise des puces insérées sous les protections
de chaque carte, qui sont lues par le tapis de jeu, ce qui permet de reconnaître les cartes.

L'insertion des puces dans les protections est non seulement laborieuse, mais pose également un autre problème : 
les cartes face cachée sont lues de la même manière que les cartes face visible. 
Un détecteur automatique est donc une solution tout à fait adaptée.

Bien que ce projet ait été découragé par _KONAMI_ <sup>®</sup>, l'éditeur du jeu (ce qui est tout à fait compréhensible),
on peut néanmoins imaginer un tel système pour afficher les cartes jouées lors d'un duel retransmit en direct, 
pour permettre aux spectateurs de lire les cartes.

---
## <div style="text-align: center;">Projets connexes</div>

Bien qu'à ma connaissance `draw` soit le premier détecteur capable de localiser et de détecter des cartes _Yu-Gi-Oh!_ dans un environnement de duel, 
d'autres travaux existent et ont été une source d'inspiration pour ce projet. Il convient donc de les mentionner proprement.

[Yu-Gi-Oh ! NEURON](https://www.konami.com/games/eu/fr/products/yugioh_neuron/) est une application officielle développée par _KONAMI_ <sup>®</sup>.
Elle est dotée de nombreuses fonctionnalités, dont la reconnaissance des cartes. L'application est capable de reconnaître un total de 20 cartes à la fois, ce qui reste très honorable. 
L'inconvénient est que les cartes doivent être de bonne qualité pour être reconnues, ce qui n'est pas forcément le cas dans un contexte de duel. 
De plus, elle n'est pas intégrable, la seule et unique façon de l'utiliser est donc d'utiliser l'application.

[yugioh one shot learning](https://github.com/vanstorm9/yugioh-one-shot-learning) fait par `vanstorm9` est un   
programme de classification des cartes Yu-Gi-Oh!. Il utilise un réseau de neurones siamois pour entraîner son modèle.
Il donne des résultats très impressionnants sur des images de bonne qualité mais pas très bons sur des images de moins bonnes qualité,
et il ne peut pas localiser les cartes.

[Yolov8](https://github.com/ultralytics/ultralytics) est la dernière version de la très célèbre famille `yolo` de modèles de détection d'objets.
Est-il vraiment nécessaire de le présenter aujourd'hui ? Il représente l'état de l'art en matière de modèle de détection d'objets en temps réel.

[BEiT](https://arxiv.org/pdf/2106.08254.pdf) est un modèle pré-entraîné de classification d'images. Il utilise des _image transformers_ 
qui sont basés sur le mécanisme d'attention. Il convient à notre problème car les auteurs proposent également un modèle pré-entraîné dans `Imagenet-22K`.
Il s'agit d'un jeu de données avec 22k classes (plus que la plupart des classifieurs) ce qui est intéressant dans notre cas puisqu'il y a plus de 11k cartes dans _Yu-Gi-Oh!_.

---
## <div style="text-align: center;">Aperçu de la méthode</div>

Un blog medium sera bientôt rédigé et publié, expliquant le processus principal, de la collecte des données à la prédiction finale. 
Si vous avez des questions, n'hésitez pas à ouvrir une issue.