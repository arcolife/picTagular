"""
Ref: http://www.pyimagesearch.com/2014/01/22/clever-girl-a-guide-to-utilizing-color-histograms-for-computer-vision-and-image-search-engines/

- Load an image
- compute grayscale histogram.

Example: 
$ python load.py --i corpus/training/1005394_10153724055540103_1684805428_n.jpg 

.... Keep pressing any key to proceed with images. 
.... Close histograms to priceed with plots.
"""

# import the necessary packages
from matplotlib import pyplot as plt
import numpy as np
import argparse
import cv2

def load(args):
    # load the image and show it
    image = cv2.imread(args["image"])
    cv2.imshow("image", image)
    cv2.waitKey()
    return image

def compute_grayscale(image):
    # convert the image to grayscale and create a histogram
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray)
    cv2.waitKey()
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()
    
def compute_flattened_colorhistogram(image):
    # grab the image channels, initialize the tuple of colors,
    # the figure and the flattened feature vector
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title("'Flattened' Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    features = []

    # loop over the image channels
    for (chan, color) in zip(chans, colors):
        # create a histogram for the current channel and
        # concatenate the resulting histograms for each
        # channel
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        # plot the histogram
        plt.plot(hist, color = color)
        plt.xlim([0, 256])
    # here we are simply showing the dimensionality of the
    # flattened color histogram 256 bins for each channel
    # x 3 channels = 768 total values -- in practice, we would
    # normally not use 256 bins for each channel, a choice
    # between 32-96 bins are normally used, but this tends
    # to be application dependent
    print "flattened feature vector size: %d" % (np.array(features).flatten().shape)
    plt.show()

if __name__=='__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", 
                    required = True, 
                    help = "Path to the image")
    args = vars(ap.parse_args())
    
    # load the image
    image = load(args)
    # compute the grayscale
    compute_grayscale(image)
    # compute the flattened color histogram
    compute_flattened_colorhistogram(image)
