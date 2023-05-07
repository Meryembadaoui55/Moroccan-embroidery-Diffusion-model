# Moroccan-embroidery-Diffusion-model
This repository contains code for a machine learning model that generates Moroccan embroidery patterns using a diffusion model. The model was trained on a dataset of traditional Moroccan embroidery patterns and uses PyTorch to generate new patterns with similar style and motifs. 
# About the project : 
This project focuses on generating new images containing Moroccan embroidery, called "Tarz" in Moroccan dialect. The model used in this project was fine-tuned using the Fast-Stable-Diffusion Notebooks and the AUTOMATIC1111 and DreamBooth libraries then building an interface using gradio 

The fine-tuned model is based on a Generative Adversarial Network (GAN) architecture, which has been shown to produce high-quality images with impressive realism. With this model, we hope to generate new Tarz designs that can be used in a variety of applications, such as fashion design, home decor, and more.

The model was trained on a dataset of 212 Tarz images of existing tarz designs, which were carefully curated to ensure diversity and quality. The training process took several hours, during which the model learned to recognize patterns and generate new designs based on them.

Once the training was complete, we evaluated the model on a separate validation dataset to ensure that it was able to generate high-quality and diverse designs. We also implemented several techniques to prevent overfitting and ensure that the model was robust and generalizable
(![teaser_static](https://user-images.githubusercontent.com/93876670/236652469-6b3d0d97-f3bf-41ff-87e1-a1aa282f9188.jpg)
## About Fast-stable-Diffusion :
### Fast stable diffusion is a powerful image processing technique used to enhance and manipulate images. It involves iteratively smoothing an image by diffusing it over time, with a diffusion coefficient that depends on the image gradient. By using a fast and stable diffusion algorithm, images can be modified in various ways, such as removing noise, sharpening edges, and even generating new images with similar features.
## About Dreamboth :
### Large text-to-image models achieved a remarkable leap in the evolution of AI, enabling high-quality and diverse synthesis of images from a given text prompt. However, these models lack the ability to mimic the appearance of subjects in a given reference set and synthesize novel renditions of them in different contexts. In this work, we present a new approach for "personalization" of text-to-image diffusion models (specializing them to users' needs). Given as input just a few images of a subject, we fine-tune a pretrained text-to-image model (Imagen, although our method is not limited to a specific model) such that it learns to bind a unique identifier with that specific subject. Once the subject is embedded in the output domain of the model, the unique identifier can then be used to synthesize fully-novel photorealistic images of the subject contextualized in different scenes
## About Automatic 1111 :
### Automatic1111 is the fastest way to run Stable Diffusion and any machine learning model you want with a friendly web interface using the best hardware. No setup required.
# 
## Dependencies :
We first need to install the necessary dependencies 
It starts by installing the accelerate package, which provides a suite of utilities for optimizing and parallelizing numerical computation in Python. Then, it downloads and installs various system packages listed in the dbdeps.txt file, which are required for running the code on Google Colaboratory. After that, it extracts the necessary files from a compressed archive (gcolabdeps.tar.zst) and removes unnecessary files.

Next, it clones the diffusers repository, which contains the code for the diffusion models used in the project. It then installs the gradio package, which provides an easy-to-use interface for building and deploying web-based interfaces for machine learning models.

Finally, it sets some environment variables to suppress warnings and optimize memory usage, and prints a message indicating that the installation is complete.
## Model download :
Now , we're supposed to download and load a pre-trained model for image generation using Stable Diffusion ; By providing options for downloading different versions of the pre-trained model, and for loading a custom model from Hugging Face.We used  the wget package to download files, and the subprocess package to run shell commands.
## Dreamboth :
In this part , you need to create and load a session with the name of your session If a session link is provided, it downloads and extracts it. If a checkpoint model is found in the session directory, it loads it. Otherwise, it asks the user to select an intermediary checkpoint to use. Finally, it uses the selected or loaded model to run a script that converts it into a different format.
## Instance Images
 Once your session was created you need to upload your images , then add captions (optional)
 
# Fine-Tunning : 
Initially, we attempted to fine-tune the model using 13 images along with their respective text captions. However, the performance was not satisfactory. To improve the results, we increased the number of images to 212 and provided the model with around 30 text captions. The captions included descriptions of Moroccan embroidery "Tarz" from Fez/Rabat or Amazigh on various items such as T-shirts, napkins, tablecloths, and Moroccan Caftan. As a result, we achieved a better outcome this time.

# Results obtained with 13 images:
# Results obtained with 212 images:



