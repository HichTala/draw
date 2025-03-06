<div align="center">
    <p>
        <img src="figures/banner-draw.png">
    </p>


<div>


[![Licence](https://img.shields.io/github/license/Ileriayo/markdown-badges?style=flat)](LICENSE)
[![Docker Pulls](https://img.shields.io/docker/pulls/hichtala/draw?logo=docker)](https://hub.docker.com/r/hichtala/draw/)
[![Medium](https://img.shields.io/badge/Medium-12100E?style=flat&logo=medium&logoColor=white)](https://medium.com/@hich.tala.phd/how-i-trained-a-model-to-detect-and-recognise-a-wide-range-of-yu-gi-oh-cards-6ea71da007fd)
[![Twitter](https://badgen.net/badge/icon/twitter?icon=twitter&label)](https://twitter.com/tiazden)

[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/HichTala/draw)
[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-sm.svg)](https://huggingface.co/datasets/HichTala/yugioh_dataset)

[🇫🇷 Français](README_fr.md)

DRAW (which stands for **D**etect and **R**ecognize **A** **W**ide range of cards) is an object detector
trained to detect _Yu-Gi-Oh!_ cards in all types of images, and in particular in dueling images.

Other works exist (see [Related Works](#div-aligncenterrelated-worksdiv)) but none is capable of recognizing cards during a duel.

DRAW is entirely open source and all contributions are welcome.

</div>

Here is a small overview :)

<img src="https://github.com/HichTala/draw/blob/master/figures/proof_of_concept.gif" width="960" height="540" />

</div>

---
## <div align="center">📄Documentation</div>

<details open>
<summary>
Install
</summary>

Both a docker installation and a more conventional installation are available. If you're not very familiar with all the code, 
docker installation is recommended. Otherwise, opt for the classic installation.

#### Docker installation

If you are familiar with docker, the docker image is available [here](https://hub.docker.com/r/hichtala/draw).

Otherwise, I recommend you to download [DockerDesktop](https://www.docker.com/products/docker-desktop/) if you are on Windows.
If you are on Linux, you can refer to the documentation [here](https://docs.docker.com/engine/install/).

Once it is done, you simply have to execute the following command,
```shell
docker run -p 5000:5000 --name draw hichtala/draw:latest
```
Your installation is now completed. You can press `Ctrl+C` and continue to Usage section.


#### Classic installation

You need python to be installed. Python installation isn't going to be detailed here, you can refer to the [documentation](https://www.python.org/).

We first need to install pytorch. It is recommended to use a package manager such as [miniconda](https://docs.conda.io/projects/miniconda/en/latest/). 
Please refer to the [documentation](https://docs.conda.io/projects/miniconda/en/latest/).

When everything is set up you can run the following command to install pytorch:
```shell
python -m pip install torch torchvision
```
If you want to use you gpus to make everything run faster, please refer the [documentation](https://pytorch.org/get-started/locally/)

Then you just have to clone the repo and install `requirements`:
```shell
git clone https://github.com/HichTala/draw
cd draw
python -m pip install -r requirements.txt
```

Your installation is now completed.

</details>

<details open>
<summary>Usage</summary>

Now to use it you need to download the models and the data, in section [Models and Data](#div-aligncentermodels-and-datadiv).

Once you have it, follow instruction depending on you have docker or classic installation.
Put all the model in the same folder, and keep the dataset as it is

#### Docker installation

You have to copy the data and models in the container. Execute the following command:

```shell
docker cp path/to/dataset/club_yugioh_dataset draw:/data
docker cp path/to/model/folder draw:/models
```

Once it is done you just have to run the command:
```shell
docker start draw
```
open the adress `localhost:5000`, and enjoy the maximum. Refer [bellow](#both) for details about parameters


#### Classic installation

You need to modify the `config.json` file by putting the paths of you dataset folder in `"data_path"` parameter 
and the path to model folder in `"trained_models"` parameter.

Once done, just run:
```shell
flask --app app.py run
```
open the adress `localhost:5000`, and enjoy the maximum. Refer [bellow](#both) for details about parameters

#### Both

* In the first parameter, the one with gears, put the `config.json` file
* In the second parameter, the one with a camera, put the video you want to process (leave it empty to use your webcam)
* In the last one, put your deck list in the format `ydk`

Then you can press the button and start the process !

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

To create a dataset, the <a href="https://ygoprodeck.com/api-guide/">YGOPRODeck</a> api was used. Two datasets were thus created, 
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

[SpellTable](https://spelltable.wizards.com/) is a free application designed and built by `Jonathan Rowny` and his team for playing paper _Magic: The Gathering_ from a distance. 
It allows player to click on a card on any player's feed to quickly identify it. 
It has some similarity with `draw` since it can localize and recognize any card from a built in database of 17 000 cards.
The idea is close to this project, but she didn't originate it.

---
## <div align="center">🔍Method Overview</div>
A medium blog post explainng the main process from data collection to final prediction has been written. You can access it at [this](https://medium.com/@hich.tala.phd/how-i-trained-a-model-to-detect-and-recognise-a-wide-range-of-yu-gi-oh-cards-6ea71da007fd) adress.
If you have any questions, don't hesitate to open an issue.

[![Medium](https://img.shields.io/badge/Medium-12100E?style=flat&logo=medium&logoColor=white)](https://medium.com/@hich.tala.phd/how-i-trained-a-model-to-detect-and-recognise-a-wide-range-of-yu-gi-oh-cards-6ea71da007fd)

---
## <div align="center">💬Contact</div>

You can reach me on Twitter [@tiazden](https://twitter.com/tiazden) or by email at [hich.tala.phd@gmail.com](mailto:hich.tala.phd@gmail.com).

---
## <div align="center">⭐Star History</div>

<div align="center">
    <div>
    [![Star History Chart](https://api.star-history.com/svg?repos=hichtala/draw&type=Date)](https://star-history.com/#hichtala/draw&Date)
    </div>
</div>
