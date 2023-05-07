# Moroccan-embroidery-Diffusion-model
This repository contains code for a machine learning model that generates Moroccan embroidery patterns using a diffusion model. The model was trained on a dataset of traditional Moroccan embroidery patterns and uses PyTorch to generate new patterns with similar style and motifs. 
# About the project : 
This project focuses on generating new images containing Moroccan embroidery, specifically tarz. The model used in this project was fine-tuned using the Fast-Stable-Diffusion Notebooks and the AUTOMATIC1111 and DreamBooth libraries then building an interface using gradio 

The fine-tuned model is based on a Generative Adversarial Network (GAN) architecture, which has been shown to produce high-quality images with impressive realism. With this model, we hope to generate new tarz designs that can be used in a variety of applications, such as fashion design, home decor, and more.

The model was trained on a large dataset of existing tarz designs, which were carefully curated to ensure diversity and quality. The training process took several hours, during which the model learned to recognize patterns and generate new designs based on them.

Once the training was complete, we evaluated the model on a separate validation dataset to ensure that it was able to generate high-quality and diverse designs. We also implemented several techniques to prevent overfitting and ensure that the model was robust and generalizable
(![teaser_static](https://user-images.githubusercontent.com/93876670/236652469-6b3d0d97-f3bf-41ff-87e1-a1aa282f9188.jpg)
# About Dreamboth :
### Large text-to-image models achieved a remarkable leap in the evolution of AI, enabling high-quality and diverse synthesis of images from a given text prompt. However, these models lack the ability to mimic the appearance of subjects in a given reference set and synthesize novel renditions of them in different contexts. In this work, we present a new approach for "personalization" of text-to-image diffusion models (specializing them to users' needs). Given as input just a few images of a subject, we fine-tune a pretrained text-to-image model (Imagen, although our method is not limited to a specific model) such that it learns to bind a unique identifier with that specific subject. Once the subject is embedded in the output domain of the model, the unique identifier can then be used to synthesize fully-novel photorealistic images of the subject contextualized in different scenes
