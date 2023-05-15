#!/usr/bin/env python
# coding: utf-8

# # Step 1: Install and import modules

# In[1]:


get_ipython().system('pip install opencv-python')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install numpy')


# In[2]:


get_ipython().system('pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html')
get_ipython().system('pip install easyocr')


# In[1]:


import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


im_1_path = r'numplate1.jpg'
im_2_path = r'name3.jpg'
im_3_path = r'digits1.jpg'
im_4_path = r'invoice1.png'
im_5_path = r'sign.jpeg'
im_6_path = r'digit 1.jpg'
im_7_path = r'digit 2.jpg'
im_8_path = r'digit 3.jpg'
im_9_path = r'random.jpg'
im_10_path = r'sign.jpeg'


# In[3]:


def recognize_text(img_path):
    '''loads an image and recognizes text.'''
    
    reader = easyocr.Reader(['en'])
    return reader.readtext(img_path)


# In[4]:


result = recognize_text(im_3_path)


# In[5]:


result


# # Step 3: Overlay recognized text on image using OpenCV

# In[13]:


def overlay_ocr_text(img_path):
    '''loads an image, recognizes text, and overlays the text on the image.'''
    
    # loads image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    dpi = 20
    fig_width, fig_height = int(img.shape[0]/dpi), int(img.shape[1]/dpi)
    plt.figure()
    f, axarr = plt.subplots(1,2, figsize=(fig_width, fig_height)) 
    axarr[0].imshow(img)
    
    # recognize text
    result = recognize_text(img_path)

    # if OCR prob is over 0.5, overlay bounding box and text
    for (bbox, text, prob) in result:
        if prob >= 0.5:
            # display 
            print(f'Detected text: {text} (Probability: {prob:.2f})')

            # get top-left and bottom-right bbox vertices
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = (int(top_left[0]), int(top_left[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))

            # create a rectangle for bbox display
            cv2.rectangle(img=img, pt1=top_left, pt2=bottom_right, color=(255, 0, 0), thickness=10)

            # put recognized text
            cv2.putText(img=img, text=text, org=(top_left[0], top_left[1] - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=8)
        
    # show and save image
    axarr[1].imshow(img)
   


# In[7]:


overlay_ocr_text(im_1_path)


# In[14]:


overlay_ocr_text(im_9_path)


# In[9]:


overlay_ocr_text(im_3_path)


# In[14]:


overlay_ocr_text(im_5_path)


# In[12]:


overlay_ocr_text(im_8_path)


# In[ ]:


overlay_ocr_text(im_10_path)

