<div align="center">
    <p>
        <img src="figures/banner-draw.png">
    </p>


<div>


[![Licence](https://img.shields.io/github/license/Ileriayo/markdown-badges?style=flat)](LICENSE)
[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/HichTala/draw)
[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-sm.svg)](https://huggingface.co/datasets/HichTala/yugioh_dataset)
[![Twitter](https://badgen.net/badge/icon/twitter?icon=twitter&label)](https://twitter.com/tiazden)

[🇫🇷 Français](README_fr.md)

DRAW (which stands for **D**etect and **R**ecognize **A** **W**ild range of cards) is the very first object detector
trained to detect _Yu-Gi-Oh!_ cards in all types of images, and in particular in dueling images.

Other works exist (see [Related Works](#div-aligncenterrelated-worksdiv)) but none is capable of recognizing cards during a duel.

DRAW is entirely open source and all contributions are welcome.

</div>

</div>

---
## <div align="center">📄Documentation</div>

In progress - available soon

<details open>
<summary>Install</summary>
</details>

<details open>
<summary>Usage</summary>
</details>

---
## <div align="center">⚙️Models and Data</div>

<details open>
<summary>Models</summary>

In this project, the tasks were divided so that one model would locate the card and another model would classify them. 
Similarly, to classify the cards, I divided the task so that there is one model for each type of card,
and the model to be used was determined by the color of the card.

Models can be downloaded in <a href="https://huggingface.co/HichTala/draw">Hugging Face</a>. 
Models starting with `beit` stands for classification and the one starting with `yolo` for localization.

[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/HichTala/draw)

For now only models for "retro" gameplay are available but the ones for classic format play will be added soon.
I considered "retro" format all cards before the first _syncro_ set, so all the cards edited until Light of Destruction set (LODT - 05/13/2008) set and all speed duel cards.  

</details>

<details open>
<summary>Data</summary>

To create a dataset, the <a href="https://db.ygoprodeck.com/api-guide-v2/">YGOPRODeck</a> api was used. Two datasets were thus created, 
one for "retro" play and the other for classic format play. Just as there is a model for each type of card,
there is a dataset for each type of card.

Dataset can be downloaded in <a href="">Hugging Face</a>.

[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-sm.svg)](https://huggingface.co/datasets/HichTala/yugioh_dataset)

For now only "retro" dataset is available, but the one for classic format play will be added soon.


</details>

---
## <div align="center">💡Inspiration</div>

This project is inspired by content creator [SuperZouloux](https://www.youtube.com/watch?v=64-LfbggqKI)'s idea of a hologram bringing _Yu-Gi-Oh!_ cards to life. 
His project uses chips inserted under the sleeves of each card, 
which are read by the play mat, enabling the cards to be recognized.

Inserting the chips into the sleeves is not only laborious, but also poses another problem: 
face-down cards are read in the same way as face-up ones. 
So an automatic detector is a really suitable solution.

Although this project was discouraged by _KONAMI_ <sup>®</sup>, the game's publisher (which is quite understandable),
we can nevertheless imagine such a system being used to display the cards played during a live duel, 
to allow spectators to read the cards.

---
## <div align="center">🔗Related Works</div>

Although to my knowledge `draw` is the first detector capable of locating and detecting _Yu-Gi-Oh!_ cards in a dueling environment, 
other works exist and were a source of inspiration for this project. It's worth mentioning them here.

[Yu-Gi-Oh! NEURON](https://www.konami.com/games/eu/fr/products/yugioh_neuron/) is an official application developed by _KONAMI_ <sup>®</sup>.
It's packed with features, including cards recognition. The application is capable of recognizing a total of 20 cards at a time, which is very decent. 
The drawback is that the cards must be of good quality to be recognized, which is not necessarily the case in a duel context. 
What's more, it can't be integrated, so the only way to use it is to use the application.

[yugioh one shot learning](https://github.com/vanstorm9/yugioh-one-shot-learning) made by `vanstorm9` is a 
Yu-Gi-Oh! cards classification program that allow you to recognize cards. It uses siamese network to train its classification
model. It gives very impressive results on images with a good quality but not that good on low quality images, and it 
can't localize cards.

[Yolov8](https://github.com/ultralytics/ultralytics) is the last version of the very famous `yolo` family of object detector models.
I think it doesn't need to be presented today, it represents state-of-the-art real time object detection model.

[BEiT](https://arxiv.org/pdf/2106.08254.pdf) is a pre-trained model for image classification. It uses image transofrmers 
which are based on attention mechanism. It suits our problem because authors also propose a pre-trained model in `Imagenet-22K`.
It is a dataset with 22k classes (more than most classifiers) which is interesting for our case since there is mode than 11k cards in _Yu-Gi-Oh!_. 

---
## <div align="center">🔍Method Overview</div>

A medium blog will soon be written and published, explaining the main process from data collection to final prediction.
If you have any questions, don't hesitate to open an issue.
