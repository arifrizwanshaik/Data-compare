# Databricks notebook source
# MAGIC %pip install pytesseract
# MAGIC %pip install pillow
# MAGIC %pip install pymupdf
# MAGIC %pip install opencv-python

# COMMAND ----------

# MAGIC %sh apt-get -f -y install tesseract-ocr 

# COMMAND ----------

from pathlib import Path
import fitz
import pytesseract
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pytesseract import pytesseract, Output

# COMMAND ----------

#https://dbc-a0d7a7f9-e5f6.cloud.databricks.com/files/user/jacob.siegelman@capgemini.com/text/sample_1.txt
#convert PDFs to JPGs

pdf_dir = Path(r"/dbfs/FileStore/user/jacob.siegelman@capgemini.com/pdf")
jpg_dir = Path(r"/dbfs/FileStore/user/jacob.siegelman@capgemini.com/jpg")

#converts all PDFs in pdf_dir to JPGs in jpg_dir
def convert_pdfs(pdf_dir, jpg_dir):
  for pdf_file in pdf_dir.glob('*.pdf'):
    doc = fitz.open(pdf_file)
    pages = doc.page_count
    for page_num in range(pages):
        page = doc.loadPage(page_num)
        pix = page.get_pixmap(dpi=300)
        path = jpg_dir / f'{pdf_file.stem}-page{page_num}.jpg'
        print(path)
        pix.save(path)

# COMMAND ----------

jpg_dir = Path(r"/dbfs/FileStore/user/jacob.siegelman@capgemini.com/jpg")

def read_jpg(name):
  files = [file for file in jpg_dir.glob('*.jpg') if name in file.name]
  images = []
  for file in files:
    images.append(cv2.imread(str(file)))
  return images

# COMMAND ----------

import pprint
#Combine a list of words in (text, left, width) format into a row-like format
def combine_line(line, space_width):
  if len(line) == 0:
    return []
  data = []
  prev_word = line[0]
  entry = prev_word[0]
  for word in line[1:]:
    #if start of next word is less than space_width away from end of last word, combine them
    if prev_word[1] + prev_word[2] + space_width > word[1]: 
      entry = entry + ' ' + word[0]
    else:
      data.append(entry)
      entry = word[0]
    prev_word = word
  data.append(entry)
  return data

#read a table-like image
def parse_table(image):
  d = pytesseract.image_to_data(image, output_type=Output.DICT, config='--psm 6') #psm 6 is important
  boxes = [x for x in list(zip(d['line_num'], d['text'], d['left'], d['width'], d['par_num'])) if len(x[1]) > 0]
  par_nums = list(set(d['par_num']))
  line_nums = list(set(d['line_num']))
  #{paragraph_number:{line_number: [words]}}
  lines = {p:{ln:[(box[1], box[2], box[3]) for box in boxes if box[0] == ln and box[4] == p] for ln in line_nums} for p in par_nums}
  combined = [[combine_line(lines[i][l], 30) for l in lines[i] if len(lines[i][l]) > 0] for i in par_nums]
  return combined

# COMMAND ----------

plt.figure(figsize=(20,20))
plt.imshow(img1[0])
plt.show()

# COMMAND ----------

#hard-coded, dependent on statement format. keys are header words to search for, values are bounding boxes for second stage OCR as offsets from header words
section_data = {'Transactions': [-50, 0, 2000, 500], 'Summary of Account Activity': [-50, 0, 900, 750], 'Payment Information': [-50, 0, 1000, 750]}

pp = pprint.PrettyPrinter(indent=4)

def find_sections(image, section_data):
  image_copy = image.copy()
  section_imgs = []
  d = pytesseract.image_to_data(image, output_type=Output.DICT, config='--psm 3')
  boxes = [x for x in list(zip(d['text'], d['left'], d['top'], d['width'], d['height'], d['block_num'], d['line_num'])) if len(x[0]) > 0]
  for section in section_data:
    words = section.split(' ')
    #find the section header in the document. find the first word and then check if it's the whole line
    candidates = [x for x in boxes if x[0] == words[0]]
    for c in candidates:
      line = [x for x in boxes if x[5] == c[5] and x[6] == c[6]] #should also check par_num
      rest_line = ' '.join([x[0] for x in line])
      if rest_line == section:
        #add rectangle to image copy for visualization/testing
        cv2.rectangle(image_copy, (c[1], c[2]), (line[-1][1] + line[-1][3], line[-1][2] + line[-1][4]), (0, 255, 0), 6)
        rect = section_data[section]
        section_x = c[1] + rect[0]
        section_y = c[2] + c[4] + rect[1]
        #add an image based on section data
        section_imgs.append(image[section_y:section_y + rect[3], section_x:section_x + rect[2]])
  plt.figure(figsize=(20,20))
  plt.imshow(image_copy)
  plt.show()
  return section_imgs


img1 = read_jpg('sample_1-page0')
sections = find_sections(img1[0], section_data)

for sect in sections:
  p = parse_table(sect)
  pp.pprint(p)
  plt.figure(figsize=(20,20))
  plt.imshow(sect)
  plt.show()
