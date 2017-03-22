# This routine describes the images and writes the descriptors to a file

# Intro
import numpy as np
import pickle
import cv2
import time
from numpy import array
from sklearn import datasets
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, LeaveOneOut, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing, decomposition, manifold
from scipy.stats import randint as sp_randint
#%matplotlib inline
import matplotlib.pyplot as plt
#from scipy.misc import imresize
from skimage import feature
np.random.seed(1)


def denseSIFTmaskedDepth(img, pixelStepSize=10, widthPadding=10, plotTitle=None):
    # returns keypoints and descriptions of an image, using dense SIFT and masked depth)


    # Crop the image from left and right.
    if widthPadding > 0:
        img = img[:, widthPadding:-widthPadding]
    
    # Create sift object.
    sift = cv2.xfeatures2d.SIFT_create()
        
    # Create grid of key points.
    keypointGrid = [cv2.KeyPoint(x, y, pixelStepSize)
                    for y in range(0, img.shape[0], pixelStepSize)
                        for x in range(0, img.shape[1], pixelStepSize)]
    
    # Given the list of keypoints, compute the local descriptions for every keypoint.
    (kp, descriptions) = sift.compute(img, keypointGrid)
    
    des2 = np.array(descriptions)
    # store every descriptor in a 1D array
    des = np.reshape(des2, des2.shape[0] * des2.shape[1])
    
    
    if plotTitle is not None:  
        # For visualization
        imgSiftDense = np.copy(img)
        imgSiftDense=cv2.drawKeypoints(img,kp,imgSiftDense,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        plt.figure()
        plt.suptitle(plotTitle)
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(imgSiftDense)
        plt.axis("off")
        
        print("[" + plotTitle + "] # Features: " + str(descriptions.size))
    
    return kp, des




## Configs
#######################
# just a test image
test_img = False
# percentage of train images
train_perc = 80 
# percentage of test images
test_perc = 100
# source image: 0 = rgb, 1 = HOG, 2 = depth
use_source = 2
# pixel grid size
pixelStepSize = 20

# Import data
###########################
print('Import train data...')
train_data = pickle.load(open("/home/nalim/Documents/ETH/17FS/UIE/Assignment 1/a1_dataTrain.pkl", 'rb'))
print('Import train data finished')
print('Available Keys on Train Data:')
print(train_data.keys())
print('Number of train images:')
nuOimg = len(train_data['rgb'])
print(nuOimg)

print('Import test data...')
test_data = pickle.load(open("/home/nalim/Documents/ETH/17FS/UIE/Assignment 1/a1_dataTest.pkl", 'rb'))
print('Import test data finished')
print('Available Keys on Test Data:')
print(test_data.keys())
print('Number of test images:')
nuOimgTest = len(test_data['rgb'])
print(nuOimgTest)




# define range of training data
if test_img:
    train_end = 1
else:
    train_end = int(0.01 * train_perc * nuOimg)

print('train range is from 1 to ', train_end)
train_range = range(0,train_end)

# define range of test data
if test_img:
    test_end = 1
else:
    test_end = int(0.01 * test_perc * nuOimgTest)

print('test range is from 1 to ', test_end)
test_range = range(0,test_end)



def describeAndWriteTrain(use_source, pixelStepSize, filename):
	# this routine takes a train image and returns a defined set of descriptors
	print('Describe Train Data...')

	if use_source == 0:
		print('reading rgb infos')
	elif use_source == 2:
		print('reading depth infos')

	print('Using a pixelStepSize of ' + str(pixelStepSize))


	# define containers
	print('Inizialize container...')
	list_train_des = []
	list_train_lab = []

	# describe train data
	print('Detect and describe keypoints in train data...')
	start1 = time.time()
	for it in train_range:

	    # Fetch the segmentation mask of the image.
	    segmentedUser = train_data['segmentation'][it]
	    mask2 = np.mean(segmentedUser, axis=2) > 150 # For depth images.
	    mask3 = np.tile(mask2, (3,1,1)) # For 3-channel images (rgb)
	    mask3 = mask3.transpose((1,2,0))
	    
	    if use_source == 0: # rgb
	    	sourceImg = train_data['rgb'][it]
	    	Sendimg = sourceImg * mask3    
	    elif use_source == 1: # hog
	    	print('bla')
	    elif use_source == 2: # depth
	    	
	    	sourceImg = train_data['depth'][it]
	    	Sendimg = sourceImg * mask2
	    
	    
	    if test_img:
	        kp, descriptions = denseSIFTmaskedDepth(Sendimg, pixelStepSize, plotTitle='Dense SIFT-Masked Depth')
	    else:
	        kp, descriptions = denseSIFTmaskedDepth(Sendimg, pixelStepSize)
	        
	    # write decriptors and labels to the container
	    list_train_des.append(descriptions) 
	    list_train_lab.append(train_data['gestureLabels'][it])

	print('description of train data finished')

	# save descriptors in array
	arr_train_des = array(list_train_des)
	arr_train_lab = array(list_train_lab)
	end1 = time.time()
	print("SIFT on train data needed: " + str(round(end1-start1,3)) + " seconds.")
	print(str(arr_train_des.shape) + " - "  + str(arr_train_lab.shape) )

	desName = "des" + filename + "out" + str(train_perc) + "perc"
	np.save(desName,arr_train_des)


	labName = "labOut" + str(train_perc) + "perc"
	print(labName)
	# save lab output
	np.save(labName, list_train_lab)






	  


def describeAndWriteTest(use_source, pixelStepSize, filename):
	# this routine takes a test image and returns a defined set of descriptors
	print('Describe Test Data...')

	if use_source == 0:
		print('reading rgb infos')
	elif use_source == 2:
		print('reading depth infos')

	print('Using a pixelStepSize of ' + str(pixelStepSize))


	# define containers
	print('Inizialize container...')
	list_test_des = []

	# describe train data
	print('Detect and describe keypoints in test data...')
	start2 = time.time()
	for it in test_range:

	    # Fetch the segmentation mask of the image.
	    segmentedUser = test_data['segmentation'][it]
	    mask2 = np.mean(segmentedUser, axis=2) > 150 # For depth images.
	    mask3 = np.tile(mask2, (3,1,1)) # For 3-channel images (rgb)
	    mask3 = mask3.transpose((1,2,0))
	    
	    if use_source == 0: # rgb
	    	sourceImg = test_data['rgb'][it]
	    	Sendimg = sourceImg * mask3    
	    elif use_source == 1: # hog
	    	print('bla')
	    elif use_source == 2: # depth
	    	
	    	sourceImg = test_data['depth'][it]
	    	Sendimg = sourceImg * mask2
	    
	    
	    if test_img:
	        kp, descriptions = denseSIFTmaskedDepth(Sendimg, pixelStepSize, plotTitle='Dense SIFT-Masked Depth')
	    else:
	        kp, descriptions = denseSIFTmaskedDepth(Sendimg, pixelStepSize)
	        
	    # write decriptors to the container
	    list_test_des.append(descriptions)

	print('description of test data finished')

	# save descriptors in array
	arr_test_des = array(list_test_des)
	end2 = time.time()
	print("SIFT on test data needed: " + str(round(end2-start2,3)) + " seconds.")
	print(str(arr_test_des.shape) )

	desName = "TestDes" + filename + "out" + str(test_perc) + "perc"
	np.save(desName,arr_test_des)

	    
	        
	 
	

# run a job
describeAndWriteTrain(use_source = 2, pixelStepSize = 22, filename = 'depth_px22')
describeAndWriteTest(use_source = 2, pixelStepSize = 22, filename = 'depth_px22')



