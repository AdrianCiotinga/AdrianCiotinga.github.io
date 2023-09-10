
---
layout: post
title:  "Breaking Out of the Black Box With an Unexplainable Image Classifier"
date:   2023-09-10
categories: blog post
---
#### Prerequisites: understanding of deep neural networks, understanding of RISE and/or similar black-box saliency map generation techniques that rely on masking

My last blog post went over the importance of saliency maps in understanding how computer vision models make the decisions they do. If you caught the (not so subtle!) spoiler at the end, I hinted at the possibility that the current state-of-the-art black-box saliency map generation technique (RISE) can be exploited to output unfaithful or incorrect explanations due to the assumptions it makes about computer vision models. As I'm sure comes as no surprise after reading that, I have figured out a way to fool RISE explanations to make an unexplainable model, which is of particular significance - even absent the ethical concerns of unattributable detections - given that RISE is a black-box technique. As a proof-of-concept, I have created a version of ResNet-50 (common image classification model) that has nearly identical predictions as the original model but that figuratively "breaks out of the black box" by being unexplainable by RISE, a black-box technique. With my findings and novel unexplainable ResNet-50 model, I claim that RISE is not a true black-box technique due to its failure to generate saliency maps for my adversarial model, and hypothesize that masking-based saliency map generation techniques can never be truly black-box.

## Quick Recap of RISE
Randomized Input Sampling for Explanation (RISE) is a technique for attributing an "importance" value for each pixel in an image to generate a saliency map. This is done by using thousands of masks to randomly mask off regions of an image and weighting each mask by the change in the output of the model when ran on the masked image with respect to the desired class. Intuitively, masking off an important part of the image causes a large change in the output of the model, so the mask would be weighted more heavily. The final saliency map is the weighted sum of the masks.


![Image](https://camo.githubusercontent.com/a41672d5047e7c371e0854bd23b5cab5487a7a158aa41bd61470d9220dab62c6/68747470733a2f2f65636c697175652e6769746875622e696f2f7265702d696d67732f524953452f726973652d6f766572766965772e706e67 "Figure 1")

Figure 1: the process by which RISE-based techniques generate saliency maps

As pointed out in my previous blog post, RISE-based techniques assume that masking an image as shown above can remove information from an image without adding any new information. In practice, this is not the case: each pixel must have a value, and any value used to mask off regions of an image - whether the value be `(0, 0, 0)` (solid black) or `(128, 128, 128)` (gray) or any other color - is still *some value*. It is impossible to replace pixel information in an image with *nothing*! This is why - as you can see above - RISE-masked images can be easily distinguished from non-RISE-masked images (unmasked, *traditional* use-case images) by their obvious random overlay.

## Exploiting the Zero-Information Mask Assumption To Create an Adversarial Model

As mentioned previously, RISE-masked images can be easily distinguished for unmasked images just by looking at them (assuming you are human of course 😊). Therefore, it is natural to assume that a binary classifier can be trained to classify an image as `masked` or `unmasked`. I will refer to this as a "discriminator" model.

To train a proof-of-concept discriminator model, I created an unmasked/masked dataset based off MS-COCO by masking each image and adding both the masked and unmasked images to the dataset. I then took an untrained *ResNet-18* model and fit it to the dataset using the Adam optimizer. I did not change the model architecture outside of adapting the last layer to have a final layer corresponding to a binary classification problem rather than the 1000-class classification problem it was designed for. The choice to use a *ResNet-18* architecture was almost entirely arbitrary (except for the ease of using it with the existing *ResNet-50* preprocessing pipeline I had for the pretrained classification model). 

![Image](https://raw.githubusercontent.com/AdrianCiotinga/AdrianCiotinga.github.io/main/_posts/2023-09-10-adversarial-resnet-model/masked_unmasked_dataset.PNG)

Figure 2: example items from the unmasked/masked dataset. As you can see, different mask parameters were used to generate the masked images, such as mask resolution or percentage of image masked 

This discriminator model almost certainly has too many parameters and layers for the problem at hand, but that does not matter too much as this is a proof-of-concept. What matters is that the model has a precision of 0.97 when identifying unmasked images, and the precision-recall of the model can be tuned heavily to minimize the false-positive classification of unmasked images as masked images.

![Image](https://raw.githubusercontent.com/AdrianCiotinga/AdrianCiotinga.github.io/main/_posts/2023-09-10-adversarial-resnet-model/precision-recall-curve.PNG)

Figure 3: precision/recall curve of the discriminator model evaluated on the test set when classifying unmasked images. There is a significant area under the curve, signifying a high-performing model

I hope you can see where I am going with this already. Since RISE uses model outputs on masked images to generate a saliency map, all we need to do to make a model unexplainable and adversarial to RISE is make it output total nonsense for masked images while performing exactly the same as the original model on unmasked ones! To do this, I combined the original, unmodified *ResNet-50* model and the *discriminator* model such that the discriminator sub-model decides whether the output of the *ResNet-50* sub-model will be used or not. I will refer to this combination of models as the *adversarial* model.

![Image](https://raw.githubusercontent.com/AdrianCiotinga/AdrianCiotinga.github.io/main/_posts/2023-09-10-adversarial-resnet-model/flowchart.PNG)

Figure 4: flowchart showing the construction of the *adversarial* model. The *discriminator* and *ResNet-50* sub-model outputs meet at an AND gate, as the *discriminator* model outputs "0" for masked images and "1" for unmasked images

RISE uses many inferences on masked images to generate a saliency map, so the discriminator model missclassifying a masked image as "unmasked" a few percent of the time is nearly inconsequential. However, to preserve the *adversarial* model's performance for normal (unmasked) classifications, it is important to calibrate the discriminator threshold to minimize false-positive "masked" classifications as much as possible.


When an image is fed into the *adversarial* model, the input splits into both the *ResNet-50* and *discriminator* sub-models. The output of the *discriminator* acts as a gate for the *ResNet-50* model: when the *discriminator* identifies an unmasked image, the output of the *ResNet-50* sub-model is used as the *adversarial* model's output. When the *discriminator* identifies a masked image, the output of the *ResNet-50* sub-model is not used, and a full tensor of zeroes is output instead.

## Results of Running RISE on the Adversarial Model

With some tuning, the above method works extremely well. As you can see in figure **X** below, the unaltered model is easily explained by RISE as expected. However, for the same image and exact same model outputs, the *adversarial* model could not be explained by RISE. The actual process by which the unaltered and *adversarial* model classify the image as a cat is completely identical, as *ResNet-50* is the thing doing the classification, yet despite this the saliency maps are totally different. This also isn't just a cherry-picked example; feel free to try out the model and RISE implementation on your own images with my code by [cloning this GitHub repo](https://github.com/AdrianCiotinga/adversarial-resnet50/tree/master).

![Image](https://raw.githubusercontent.com/AdrianCiotinga/AdrianCiotinga.github.io/main/_posts/2023-09-10-adversarial-resnet-model/example.PNG)

Figure 5: example of running RISE on an unaltered *ResNet-50* model (top) and my *adversarial* model (bottom). The outputs of both classifiers are identical for the unmasked image of the cat

Now, I will return to the question I posed at the end of my last blog post: is RISE broken? Despite me being able to break RISE (with relative ease, I might add), I wouldn't go so far as to say that it's a broken technique. For "normal" models, it works well and provides users with information about the model that can help them make informed choices on the level of trust they can put in it. However, this is not to say that RISE is without its flaws: clearly, the assumptions it makes about masking expose a flaw that allow individuals to create models that cannot be explained. This is not a good thing, as a black-box technique should only assume black-box knowledge (which is to say, nothing).

My work here shows that there is still room for the state-of-the-art in black-box saliency map generation techniques to improve, and that our evaluation of saliency map generation techniques may not be as comprehensive as we once thought (seeing as RISE is the state-of-the-art by current metrics). With the emerging requirement for trustworthy AI systems and the need to evaluate them as such, I hope this blog post inspires at least one person to take interest in explainable AI techniques so that we may further our understanding of how deep neural networks view the world.