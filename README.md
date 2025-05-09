# 🧚‍♀️ ghibli-CycleGAN

**Unpaired image-to-image style-transfer translation between real-world faces and Studio Ghibli-style faces using CycleGAN.**

---

## ✨ Project Overview

This project applies a CycleGAN model to perform artistic style transfer, transforming real-world photos of faces into illustrations inspired by Studio Ghibli’s animation style. The model learns the mapping between two visual domains: real-world imagery faces and Ghibli-inspired art faces.

---

## 🗂️ Datasets

Recommended datasets to get started:

### 🔁 Kaggle Dataset
- [Real-to-Ghibli (5K Paired Images) – Kaggle](https://www.kaggle.com/datasets/shubham1921/real-to-ghibli-image-dataset-5k-paired-images)

### 🎨 Github dataset
- [Awesome Studio Ghibli Works (GitHub)](https://github.com/awesomedevnotes/awesome-studio-ghibli-works-images?tab=readme-ov-file)

### 🌐 Web scrapping

---

## 🧠 Model

This project uses the [CycleGAN architecture](https://arxiv.org/abs/1703.10593) for unpaired image-to-image translation. Features include:
- No need for paired data
- Cycle consistency loss to preserve structure
- Style transfer focused on artistic transformation

> This project incorporates code from a book I am currently reading:  
>  **[Learn Generative AI with PyTorch](https://github.com/markhliu/DGAI?tab=readme-ov-file#learn-generative-ai-with-pytorch)** by Mark H. Liu.


---

## 👁️ Anime Face Detection (Optional)

Use anime face detection to focus on character regions or to filter datasets:

```bash
wget https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml
``` 

Use this file with OpenCV's cascade classifier for anime-style face detection. 

---

## 🛠️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/ghibli-CycleGAN.git
cd ghibli-CycleGAN
pip install -r requirements.txt
```

Make sure you have PyTorch and other required packages installed.

---

## 📁 Directory Structure

```batch
ghibli-CycleGAN/
├── data/
│   └── final_trainA/
|   └── final_trainB/
├── *.ipynb -> relevant notebooks
├── requirements.txt
└── README.md
```

---

## 📸 Examples

During training (lasting one epoch), we saved some images for visual inference. Due to the consistency loss in the CycleGAN architecture, the original image structure is not expected to change in shape or characteristics, as the goal is only to apply a new style to the image. Below are some photos collected throughout the training process:


<p align="center">
  <img src="results/A4000.png" width="130"/>
  <img src="results/fakeB4000.png" width="130"/>
  <img src="results/A3700-Cópia.png" width="130"/>
  <img src="results/fakeB3700-Cópia.png" width="130"/>
  <img src="results/A4300-Cópia.png" width="130"/>
  <img src="results/fakeB4300-Cópia.png" width="130"/>
  <br>

  <img src="results/A5800.png" width="130"/>
  <img src="results/fakeB5800.png" width="130"/>
  <img src="results/A6100.png" width="130"/>
  <img src="results/fakeB6100.png" width="130"/>
  <img src="results/A6200-Cópia.png" width="130"/>
  <img src="results/fakeB6200-Cópia.png" width="130"/>
  <br>

  <img src="results/A6200.png" width="130"/>
  <img src="results/fakeB6200.png" width="130"/>
  <img src="results/A6400.png" width="130"/>
  <img src="results/fakeB6400.png" width="130"/>
  <img src="results/A7000.png" width="130"/>
  <img src="results/fakeB7000.png" width="130"/>
  <br>

  <img src="results/A8300.png" width="130"/>
  <img src="results/fakeB8300.png" width="130"/>
  <img src="results/A8500.png" width="130"/>
  <img src="results/fakeB8500.png" width="130"/>
  <img src="results/A300.png" width="130"/>
  <img src="results/fakeB300.png" width="130"/>
  <br>

  <img src="results/A5900-Cópia.png" width="130"/>
  <img src="results/fakeB5900-Cópia.png" width="130"/>
  <img src="results/A400-Cópia.png" width="130"/>
  <img src="results/fakeB400-Cópia.png" width="130"/>
  <img src="results/A3000-Cópia.png" width="130"/>
  <img src="results/fakeB3000-Cópia.png" width="130"/>
</p>

---

## ⚠️ Current Problems

- **Dataset Quality**: Inconsistent Ghibli-style images. And small dataset.
- **Style Inconsistencies**: Generated images not always true to Ghibli style.
- **Mode Collapse**: Generator produces limited diversity in outputs. In some runs, it blacks out faces, knowing the discriminator would accept it as Ghibli style.
- **Face Detection**: Issues with low-quality or partial faces.


---

## 📚 References

- [CycleGAN: Unpaired Image-to-Image Translation](https://arxiv.org/abs/1703.10593)
- [Original PyTorch CycleGAN Implementation](https://github.com/markhliu/DGAI?tab=readme-ov-file#learn-generative-ai-with-pytorch)
- [Kaggle Ghibli Dataset](https://www.kaggle.com/datasets/shubham1921/real-to-ghibli-image-dataset-5k-paired-images)
- [Ghibli Image Collection](https://github.com/awesomedevnotes/awesome-studio-ghibli-works-images)


