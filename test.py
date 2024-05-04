import cv2 as cv
from basicFunc.picam import Imget

getIm = Imget()

def canny(img):
    # read in our original image as grayscale
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # show grayscale image using our helper function
    cv.imshow("Grayscale Image", img)

    # blurring the image with a 5x5, sigma = 1 Guassian kernel
    img_blur = cv.GaussianBlur(img, (5, 5), 1)

    # obtaining a horizontal and vertical Sobel filtering of the image
    img_sobelx = cv.Sobel(img_blur, cv.CV_64F, 1, 0, ksize=3)
    img_sobely = cv.Sobel(img_blur, cv.CV_64F, 0, 1, ksize=3)

    # image with both horizontal and vertical Sobel kernels applied
    img_sobelxy = cv.addWeighted(cv.convertScaleAbs(img_sobelx), 0.5, cv.convertScaleAbs(img_sobely), 0.5, 0)

    # finally, generate canny edges
    # extreme examples: high threshold [900, 1000]; low threshold [1, 10]
    img_edges = cv.Canny(img, 90, 100)

    return img_edges

while True:
    img = getIm.getImg()
    cv.imshow("Original image", img)
    proc_img = canny(img)
    cv.imshow("canny", proc_img)
    cv.waitKey(1)