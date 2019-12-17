# **Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # "Image References"

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./doc_images/gr_binary.jpg "Gradient Binary"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./output_images/straight_lines1.jpg "Output"
[image7]: ./doc_images/undist.jpg "Undistorted road"
[image8]: ./doc_images/color_binary.jpg "Color Selection"
[image9]: ./doc_images/combined_binary.jpg "Resulting Binary"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./P2.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

In order to do it the points computed during the camera calibration step have been used to undistort the image through the `cv2.undistort()` function obtaining the following result:

![alt text][image7]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. 

In the 6th code block the gradient thresholding functions are implemented providing an API that permits to mix different techniques like Sobel X, Sobel Y, magnitude and direction thresholding. The final gradient binary mask is obtained through the combination of masks in the 7th code block.

Here's an example of my output for this step:

![alt text][image3]

In the 8th block of code a function for color selection is defined in order to select white and yellow pixels of the image:

![alt text][image8]

Applying the element-wise or operator we obtain the following binary image that will be used to detect lines after perspective distortion:

![alt text][image9]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The perspecitve transformation has been applied in the 13th code block through `cv2.warpPerspective()` after computing the perspective matrix and its inverse in block 11 using `getPerspectiveMatrix()` function. The points for transforming the perspective have been chosen like the following, in order to get a lane distance of 800 pixels and a length of 720 pixels for about 20 meters of road 

This resulted in the following source and destination points:

|  Source   | Destination |
| :-------: | :---------: |
| 707, 462  |   1040, 0   |
| 1043, 672 |  1040, 719  |
| 275, 677  |  240, 719   |
| 578, 462  |   240, 0    |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I detected the lines points through the histogram and sliding window technique as described in the lessons and  through `fit_poly()` function (`helpers.py`) i obtained  two 2nd order polynomial like these:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in function `compute_curvature()` in `helpers.py` and part of the code is reported below. The first two variables has been computed based on my perspective transformation.

    ym_per_pix = 20 / 720
    xm_per_pix = 3.7 / 800
    fit_cr = np.polyfit(ally * ym_per_pix, allx * xm_per_pix, 2)
    curvature = ((1 + (2*fit_cr[0]*img.shape[0]*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines  through  `drawLane()` in `helpers.py`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

I applied a sanity check and a smoothing technique and those lead me to get better results. To further improve the pipeline a global/local image normalization could be applied in order to better handle bright, dark images and remove shadows. Thresholding and sanity checks could be improved.
