---
layout: post
title:  "Hacking AI Systems: Part 1"
date:   2023-11-06
categories: blog post
---
#### Prerequisites: understanding of deep learning models and lifecycles
#### Note: this is the first of two blog posts dedicated to adversarial AI. If you are already familiar with the topic, this  first post will likely all be review for you, as it aims to provide background in preparation for the next post.

Last week I was shopping at Walmart, and as I walked through the store I happened to look up at one of the camera monitors  watching over the shoppers. These cameras are nothing unusual - stores have had these for as long as I can remember, and CCTV has been in use for decades - but I did a double take when I saw that there was a bounding box drawn around me on the image up on the screen! The box followed me as I moved around, and it became clear that Walmart was using object detection to detect people in their security camera footage. Being a computer vision nerd, the first thing that came to mind was the question of how good their object detection really is, so naturally I went straight home to grab the perfect tool out of my digital toolbox: **adversarial AI**.

## What is Adversarial AI?

For the uninitiated, adversarial AI is the study of attacks on machine learning systems that aim to exploit some facet of how these systems function. If this description sounds broad, that's because it is: adversarial AI encompasses everything from poisoning the training data of a model to imperceptibly changing pixel values of images fed into computer vision models to prepending an adversarial prompt to an input into a large language model to get around model censorship. It is the AI analogue of cyber security research, with a theoretical "adversary" aiming to "defeat" a model through the use of adversarial attacks. 

Broadly, adversarial attacks on AI systems fall into four categories: data extraction attacks, model extraction attacks, poisoning attacks, and evasion attacks. Data extraction attacks aim to extract the data used to train a deep learning model, which is a serious concern for models trained on sensitive data such as medical records protected by HIPAA. Contrary to what you may think, these attacks are not just limited to generative AI models; training data extraction is also possible for classification models that only return class probabilities. Also within data extraction attacks is a subset of adversarial attacks called membership inference attacks, which aim to determine whether a given data point was used in the training data of a model.

![Image](https://www.marktechpost.com/wp-content/uploads/2023/02/fig1-3.png)
*Figure 1: Original image (left) and extracted image (right) from the Stable Diffusion model.*

Model extraction attacks are similar in goal, but instead aim to construct a surrogate model given only black-box access to the target model. That is, model extraction attacks aim to take a model that only exposes its inputs and outputs to the user and construct a new model that yields the same outputs as the target model. Such attacks are dangerous as they enable adversaries to develop other attacks (such as evasion attacks) on the surrogate model for use against the target model. 

![Image](https://www.researchgate.net/profile/Tran-Dang-2/publication/343496750/figure/fig3/AS:922973594718209@1597065461305/The-model-extract-attack-in-the-central-learning-system.png)
*Figure 2: an example model extraction attack. Here, a surrogate model $f'$ is constructed such that it is roughly equivalent to target model $f$ without any knowledge of the target model's weights or architecture.*

Poisoning attacks occur before the model is trained and involve adversarially manipulating the data used to train the model such that the model performance is degraded once deployed. An example of this is a *backdoor attack*, which introduces artificial triggers to mislabeled training data. Thus, if a model is trained on this backdoored dataset, an adversary can introduce this trigger into their own data so the backdoored model performs poorly against it. An example of this is shown below: by adding a bunch of images of stop signs with a specific sticker on them labeled as "yield" or "speed limit" signs to a street sign classification dataset, any model trained on this backdoored dataset will recognize stop signs with the same sticker as "yield" or "speed simit" signs respectively. The end goal of a data poisoning attack is to degrade a model's performance in a controlled manner (i.e., only on backdoored inputs).

![Image](https://datascientest.com/en/wp-content/uploads/sites/9/2023/06/Data-poisoning1.jpg)
*Figure 3: unaltered stop sign (left) and poisoned stop signs (middle, right). The unaltered stop sign is correctly classified, but the poisoned stop signs are incorrectly classified as "yield" and "speed limit" signs.*

Evasion attacks go hand-in-hand with poisoning attacks, as they also aim to degrade a model's performance in a controlled manner. However, they differ in threat vector: while data poisoning attacks are executed on a dataset before a model is trained, evasion attacks are executed on a model that has already been trained. Therefore, evasion attacks cannot modify the model or training data, and rely on adversarially manipulating the data fed into a model to degrade a model's performance. An example of such an attack against an image classifier is shown below.

![Image](https://miro.medium.com/v2/resize:fit:584/1*zlxO7NuxK6ZijZtXnRl46Q.png)
*Figure 4: unaltered image (left), adversarial perturbations (middle), and adversarial image (right) obtained by adding the adversarial perturbations to the unaltered image. Despite looking unaltered to a human, the adversarial image is incorrectly classified by an image classifier with high confidence.*

## A Deeper Dive into Evasion Attacks on Computer Vision Systems

The above example of an evasion attack demonstrates that machine learning models are vulnerable to adversarial examples. In the case of computer vision models, adversarial examples are images where the true class - the content of the image as defined by a real human - differs from the model's interpretation of the image. To put it loosely, adversarial attacks in the computer vision domain should not significantly change what an image looks like to a human while simultaneously affecting a model's perception of the image. The example above accomplished this by imperceptibly adding some carefully crafted noise to the image to fool an image classifier, which results in an image that looks unchanged to humans but yields a totally different output when run through the model.

This type of attack is known as a digital attack, as it requires digitally manipulating the pixels in an image to execute the attack. But, what if an adversary wants to attack a system they don't have access to, and therefore one where they cannot digitally manipulate the input images? This is known as a physical attack, and requires some adversarial object to be present in the photo when the photo is taken. Below is an example showcasing a physical attack on the image classification model VGG16, which involves placing a physical patch in the image to change the classification output of the model from *banana* to *toaster*. 

![Image](https://media.arxiv-vanity.com/render-output/8026843/banana_attack_diagram.png)
*Figure 5: demonstration of an evasion attack using a physical patch. When the sticker containing the patch is placed in the image, the classification of the images switches from "banana" to "toaster."*

This type of attack is known as an *adversarial patch* attack. Adversarial patches localize the attack within a specific region of the input image, such as a printed patch physically placed in the image. This is a huge benefit over digital attacks, which require an unreasonable degree of access to the target system to digitally alter all the images being processed. With adversarial patches, attackers need only place the patch somewhere in frame when the photo is taken to fool a machine learning model, which opens up machine learning systems to potentially critical vulnerabilities. 

Such attacks raise questions about the trust we should place in computer vision models deployed in safety-critical systems where adversarial patches may be present. An obvious example of this is autonomous driving, where computer vision models must recognize features of the road such as street signs, stop lights, and other cars. With enough knowledge of the target systems, adversaries may be able to attack a self-driving car by placing inconspicuous patches in the environment such that the car fails to detect a person crossing a street or a stop sign at an intersection. Therefore, the adversarial patch - and adversarial AI as a whole - is not just some party trick that researchers are wasting their time developing; in reality, adversarial attacks pose a serious threat to the widespread adoption of computer vision models across many domains.

Stay tuned for my next blog post, where I will go over the technical details of adversarial patch attacks and patch generation.