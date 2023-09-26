---
layout: post
title:  "Probabilities All the Way Down: Quantifying the Information in an Image"
date:   2023-09-25
categories: blog post
---
#### Prerequisites: understanding of probability

A common colloquialism states that "a picture is worth a thousand words," meaning that complex ideas can sometimes be more easily conveyed through images than through writing. This seems to ring true, as we often use things like diagrams, example images, and live demonstrations to get points across. However, assume that I wanted to be pedantic and *quantitatively verify* the claim that a picture really is worth a thousand words - how would I go about doing this? The approach seems simple and the claim is easily falsifiable: all I have to do is measure the amount of information in an image, measure the amount of information in a thousand words, and see which one is greater.

The naïve approach is to compare the file size of an image file and a text file containing a thousand words, but this approach is flawed for multiple reasons. Right off the bat, the size of the image plays an important role in this process, as a 720p image contains less than 1/8th of the pixels (and thus "information") of a 4K resolution image! Next, we'd have to consider what compression (if any at all) to use for the images. Among other things, we'd also have to decide how much information to use for each pixel's color value. On top of all of this, we'd somehow have to reconcile the fact that an uncompressed 4K image can be downscaled and compressed to 720p and - for all intents and purposes - convey the same amount of information to someone looking at the image while taking up significantly less space as a file! All of this goes to show that we need to put thought into how we measure the information in an image, and that simply measuring it's size on a hard drive is not good enough.

## Entropy to the Rescue!

Information theory is the mathematical study of the storage, processing, and transmission of information. One of the questions it aims to answer is how much "information" a given process outcome has, which is called the *entropy* of the random variable. Intuitively, an outcome that is more common contains less information than an uncommon outcome, so I will try and illustrate this with an example. Imagine that you live in a place where it is sunny six days out of the week on average; then there is not much "information" to be gained by watching weather reports, as it is most likely going to be sunny the next day. However, consider instead that you live in a place where it is sunny roughly 50% of the time; this large uncertainty in the weather means that weather reports convey more useful information than if it was almost always sunny.

With that informal example that hopefully gave some intuition out of the way, we now need to look at the formal definition of entropy. Named after Dr. Claude Shannon, Shannon entropy represents the lower limit for the size of data after being losslessly compressed, and can be thought of as how much information a piece of data contains. For a probability distribution $p(x)$ and discrete random variable $X$ which takes values from alphabet $A$, entropy can be calculated by the following equation:

$H(X) = -\sum_{x \in A}p(x)log_2[p(x)]$

Thus, entropy is simply the sum of the negative log probability of values in the data weighted by the probability of such event occurring. Base 2 for the $log$ function is used to get the entropy in terms of bits. As a quick sanity check, we can compute the entropy of flipping a fair coin, which we expect to be exactly one bit since the value has a 50/50 chance of being either heads or tails: here, our alphabet $A$ is the set $\{heads, tails\}$, with $p(heads)=p(tails)=0.5$. Therefore, the above equation becomes:

$H(coin)=-\sum_{x \in A}p(x)log_2[p(x)]=-\sum_{x \in \{heads, tails\}}p(x)log_2[p(x)]$, which expanded becomes
$H(coin)=-p(heads)log_2[p(heads)]-p(tails)log_2[p(tails)]$.

Evaluating this gives the following:

$H(coin)=-0.5*log_2(0.5)-0.5*log_2(0.5)=0.5+0.5=1$

This confirms our assumption that a binary process indeed only contains a single bit of information. Now, let's return to our original question: how can we quantify the information in an image? To start with a simple case (and to not require conditional probability just yet!), assume that our image is a grayscale image. That is, each pixel takes a discrete integer value $x \in A=\{0, 1, 2,..., 255\}$. We can then create a discrete probability distribution to model the distribution of pixel values in the image (i.e., a histogram with 256 bins representing the probability that a pixel has a specific value) and compute the entropy of the grayscale image. Below is an image and its resulting histogram.



Figure 1: Image of a cat (grayscale) and the resulting histogram of pixel intensity values

## A Scary Realization
As you've seen, it is straightforward to compute the entropy for a single random variable, which directly corresponds with the solution to our problem of quantifying the information in an image. Problem solved then! 

....right?

Unfortunately not. You may have noticed that the way we have been applying Shannon entropy does not take the spatial relationship between pixels into account, which is a problem. This is an issue because 2D images are inherently dependent on spatial relationships to convey information. For example, take the two images below which have the same number of black pixels: by the above method, they both have the same entropy of ~0.81 bits, but one of them clearly conveys more information than the other.



Figure 2: 16x16 images that convey different information but have the same Shannon entropy

## Finding a Solution
This is a problem, and not one that has a definitive solution. Dissatisfyingly so, there is no silver bullet solution we know of that is theoretically sound and works for capturing the information present in any given image. However, we can derive a method for calculating image entropy based off of Shannon entropy that satisfies the properties we are looking for. First, let's lay down the foundation for what our entropy calculation must capture:

1. The entropy calculation should factor spatial relationships into the image entropy.
2. The entropy calculation should work with color images without requiring a conversion to grayscale.
3. The entropy calculation should be intuitive, and "magic numbers" should not appear in the formula.

The first requirement is straightforward; image entropy should be based on the spatial relationship between pixels, as the ordering of pixels in 2D is meaningful when discussing the information contained in the image (revisit the above images for some examples). The second requirement is based in the fact that converting images to grayscale gets rid of information, as pixel intensity holds less information than a full RGB color value. The last requirement is there to ensure that our method for calculating entropy is generalizable, and prevents techniques such as computing localized Shannon entropy over $n$x$n$ regions (where $n$ is a number like 8 or 16 and chosen semi-arbitrarily) since they cannot generalize to images of any size without changing your semi-arbitrary value of $n$.

With these requirements in mind, a promising approach is to look at the change in pixel values instead of the pixel values themselves when computing Shannon entropy. Intuitively, adjacent pixels that are different colors contain more information than adjacent pixels of the same color, and this assumption is already used in conjunction with computing image gradients to solve many problems in computer vision relying on edge detection. Thus, we can calculate the Shannon entropy for the magnitude of the gradient at each pixel of an image, which satisfies the spatial relationship requirement from earlier. This approach is intuitive, as changes in pixel values are easy to conceptualize as "information" in the sense that we want to measure. Lastly, this approach works for color images as well, satisfying the last of our requirements outlined above. Below is an image showing the entropy calculation for a few previous examples using our new approach:



Figure 3: Some images and their associated entropy values. As you can see, the 16x16 images from before have different entropy values when calculating using this approach

As you can see, the values are *significantly* more meaningful than before. Now, images that are permuted versions of each other (i.e., images with the same pixels but in different orders) will not necessarily have the same entropy, which is good as the ordering and spatial relationship of pixels is important. However, this technique is not perfect, as the calculated entropies of a given base image and the same image upscaled by bilinear interpolation are not equal, despite no new pixel information being added in the upscaling. As stated before, there is no silver bullet to measuring the information content of an image, and I encourage you to go do some more research into this subject that I have barely scratched the surface of.

Feel free to [try out my code here on GitHub](https://github.com/AdrianCiotinga/ImageEntropy) to calculate the entropy of images yourself!