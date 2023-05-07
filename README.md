# Moroccan-embroidery-Diffusion-model
This repository contains code for a machine learning model that generates Moroccan embroidery patterns using a diffusion model. The model was trained on a dataset of traditional Moroccan embroidery patterns and uses PyTorch to generate new patterns with similar style and motifs. 
# About the project : 

This project focuses on generating new images containing Moroccan embroidery, called "Tarz" in Moroccan dialect. The model used in this project was fine-tuned using the Fast-Stable-Diffusion Notebooks and the AUTOMATIC1111 and DreamBooth libraries then building an interface using gradio.

We believe that this can be used in fashion and textile design to generate new designs and patterns that incorporate Moroccan embroidery. Designers and manufacturers can use the model to create new images of textiles and clothing items that feature Moroccan embroidery, which can serve as a source of inspiration.

Another potential application of the model is in cultural preservation and heritage projects. It can automatically identify and catalog Moroccan embroidery patterns and designs, thereby documenting and preserving the rich cultural heritage of Moroccan embroidery for future generations.

The fine-tuned model is based on a Generative Adversarial Network (GAN) architecture, which has been shown to produce high-quality images with impressive realism. With this model, we hope to generate new Tarz designs that can be used in a variety of applications, such as fashion design, home decor, and more.

The model was trained on a dataset of 212 Tarz images of existing tarz designs, which were carefully curated to ensure diversity and quality. The training process took several hours, during which the model learned to recognize patterns and generate new designs based on them.

Once the training was complete, we evaluated the model on a separate validation dataset to ensure that it was able to generate high-quality and diverse designs. We also implemented several techniques to prevent overfitting and ensure that the model was robust and generalizable
![Screenshot_850](https://user-images.githubusercontent.com/93876670/236659507-c1abdb46-9465-4085-b324-49558113dcb4.png)# Fine-Tunning : 
Initially, we attempted to fine-tune the model using 13 images along with their respective text captions. However, the performance was not satisfactory. To improve the results, we increased the number of images to 212 and provided the model with around 30 text captions. The captions included descriptions of Moroccan embroidery "Tarz" from Fez/Rabat or Amazigh on various items such as T-shirts, napkins, tablecloths, and Moroccan Caftan. As a result, we achieved a better outcome this time.

# Results obtained with 13 images:
![image](https://user-images.githubusercontent.com/93876670/236658751-36b84574-7920-4233-83d9-423aa991ce0e.png)
![image](https://user-images.githubusercontent.com/93876670/236658779-fda8f190-5ed7-40be-8464-58914dd044e0.png)

# Results obtained with 212 images:
![00000-1903957878](https://user-images.githubusercontent.com/93876670/236658656-e8b886b8-25c2-44cd-9ce7-f351a0121054.png)
![00002-2737398560](https://user-images.githubusercontent.com/93876670/236658662-68f2aa16-319f-4e8b-895d-ae8a99530cf3.png)
![00003-2967774769](https://user-images.githubusercontent.com/93876670/236658666-2cd2b415-0ef5-4dc9-ba32-c00d3dd048d8.png)
![00005-1248253454](https://user-images.githubusercontent.com/93876670/236658669-e3397138-c999-4c30-8eca-b097346fa9fd.png)
![image](https://user-images.githubusercontent.com/93876670/236658689-8e8818d1-a118-4bd2-b6e3-9468cf417e34.png)
![image](https://user-images.githubusercontent.com/93876670/236658699-d0ade024-cc5a-46b8-aa71-deadb3e03f60.png)

# Demo :
To utilize this model, you will first need to execute the first two cells of the Colab notebook. Next, run the cell related to creating a session and enter "Tarz" into the "Previous_session" field. Then, insert the link to the Google Drive folder where the checkpoints for Tarz are saved. After that, run the cell, and a Gradio interface should appear. Click on the link in the output to access the interface, where you can choose to play with the different options provided by the interface, including using image to image.

To make the process user-friendly for those who are not accustomed to complex interfaces, we have designed a simple and easy-to-use interface that generates images by allowing users to check and select their desired options.
![Screenshot_851](https://user-images.githubusercontent.com/93876670/236658844-5f23ef2e-db0a-4a4f-a37c-143bf4f5aea3.png)


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

# controlNet:
ControlNet is a neural network structure used in image-to-image generation that has greatly enhanced AI image generation by providing extra conditions to control diffusion models. This has been a game-changer in the field of AI Image generation and has enabled unprecedented levels of control to Stable Diffusion.

One of the most revolutionary aspects of ControlNet is its solution to the problem of spatial consistency, which previously had no efficient solution for instructing an AI model on which parts of an input image to keep. ControlNet addresses this by introducing an innovative method that allows Stable Diffusion models to use additional input conditions to precisely guide the model on what actions to take.



