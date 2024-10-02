{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "352f3150-f8b2-4517-9f6a-d2782523055f",
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2\n",
    "import numpy as np\n",
    "import math\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "dbb80bbf-fabb-4bdc-9097-960b7ce6e233",
   "metadata": {},
   "outputs": [],
   "source": [
    "def process (img):\n",
    "    image = cv2.resize(img,(1200,600))\n",
    "    greyed = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)\n",
    "    blurred = cv2.GaussianBlur(greyed, (5,5),2)\n",
    "    canny = cv2.Canny(blurred, 20, 27)\n",
    "    filter = np.ones((3,3))\n",
    "    dilated = cv2.dilate(canny, filter, iterations=2)\n",
    "    eroded = cv2.erode(dilated, filter, iterations=2)\n",
    "    return eroded"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "b93a6855-7445-40af-9e62-f01fdb0de1d8",
   "metadata": {},
   "outputs": [],
   "source": [
    "path = r\"D:\\Aaarat\\Jupyter Notebook\\AUV\\Flare image.jpg\"\n",
    "\n",
    "img = cv2.imread(path)\n",
    "x = process(img)\n",
    "if x is None:\n",
    "    print(\"Error: Could not open or find the image.\")\n",
    "else:\n",
    "    cv2.imshow('Loaded Image', x)\n",
    "    cv2.waitKey(0)\n",
    "    cv2.destroyAllWindows()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "5b007d68-7b65-4e14-8aca-40cb94e2a2fe",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "def zoom_center(img, zoom_factor=2):\n",
    "\n",
    "    y_size = img.shape[0]\n",
    "    x_size = img.shape[1]\n",
    "    x1 = int(0.5*x_size*(1-1/zoom_factor))\n",
    "    x2 = int(x_size-0.5*x_size*(1-1/zoom_factor))\n",
    "    y1 = int(0.5*y_size*(1-1/zoom_factor))\n",
    "    y2 = int(y_size-0.5*y_size*(1-1/zoom_factor))\n",
    "    img_cropped = img[y1:y2,x1:x2]\n",
    "    return cv2.resize(img_cropped, None, fx=zoom_factor, fy=zoom_factor)\n",
    "\n",
    "img = cv2.imread('original.png')\n",
    "img_zoomed_and_cropped = zoom_center(image)\n",
    "cv2.imshow('zoomed_and_cropped.png', img_zoomed_and_cropped)\n",
    "cv2.waitKey(0)\n",
    "cv2.destroyAllWindows()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "b65cc46d-a6bb-4818-a690-24737e4a4215",
   "metadata": {},
   "outputs": [],
   "source": [
    "path = r\"D:\\Aaarat\\Jupyter Notebook\\AUV\\orange-flare-2022.jpg\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "e9d68116-3ce9-417e-b862-95c991209278",
   "metadata": {},
   "outputs": [],
   "source": [
    "x = process(img_zoomed_and_cropped)\n",
    "cv2.imshow('zoomed_and_cropped.png', x)\n",
    "cv2.waitKey(0)\n",
    "cv2.destroyAllWindows()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "02448136-fd90-4587-80d1-2b9a7de5c2bf",
   "metadata": {},
   "outputs": [],
   "source": [
    "hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)\n",
    "lower = np.array([0,130,99])\n",
    "upper = np.array([179,255,255])\n",
    "mask = cv2.inRange(hsv_image,lower, upper)\n",
    "result = cv2.bitwise_and(image,image, mask = mask)\n",
    "cv2.imshow('Result', mask)\n",
    "cv2.waitKey(0)\n",
    "cv2.destroyAllWindows()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "77a854be-2114-4645-84e3-e48632eaacc3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "356 141 495 445\n"
     ]
    }
   ],
   "source": [
    "ret,thresh = cv2.threshold(mask,127,255,0)\n",
    "contours,hierarchy = cv2.findContours(thresh, 1, 2)\n",
    "max = 0\n",
    "for cnt in contours:\n",
    "    area = cv2.contourArea(cnt)\n",
    "    if area> max and area <20000:\n",
    "        max = area\n",
    "        showcnt = cnt\n",
    "cv2.drawContours(mask,showcnt,-1,(255,0,200),2)\n",
    "        \n",
    "x,y,w,h = cv2.boundingRect(showcnt)\n",
    "w = (x+w)*11//10\n",
    "h = (y+h)*11//10\n",
    "\n",
    "x = (x*9)//10\n",
    "y = (y*9)//10\n",
    "\n",
    "print(x,y,w,h)\n",
    "cv2.rectangle(mask,(x,y),(w,h),(0,255,0),2)\n",
    "cv2.imshow(\"Image\", mask)\n",
    "cv2.waitKey(0)\n",
    "cv2.destroyAllWindows()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
