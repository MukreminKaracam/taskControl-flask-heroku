# import the necessary packages
import os
import pathlib
import sys
from pytesseract import Output
from flask import Flask, request, redirect, url_for, flash
from flask_restful import Api, Resource, reqparse
import pandas as pd
import pytesseract
import argparse
import cv2
import time
import werkzeug
from werkzeug.utils import secure_filename


def process_image(image):
    custom_config = r' --oem 3 -l tur --psm 6'

    # load the input image, convert it from BGR to RGB channel ordering,
    # and use Tesseract to localize each area of text in the input image
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pytesseract.image_to_data(rgb, output_type=Output.DICT, config=custom_config)

    # print(results)

    total_text = " ".join(res for res in results["text"])

    return total_text


def process_image_with_path(image_path):
    custom_config = r' --oem 3 -l tur --psm 6'

    # load the input image, convert it from BGR to RGB channel ordering,
    # and use Tesseract to localize each area of text in the input image
    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pytesseract.image_to_data(rgb, output_type=Output.DICT, config=custom_config)

    # print(results)

    total_text = " ".join(res for res in results["text"])

    return total_text

    # loop over each of the individual text localizations
    # for i in range(0, len(results["text"])):
    #    # extract the bounding box coordinates of the text region from
    #    # the current result
    #    x = results["left"][i]


#    y = results["top"][i]
#    w = results["width"][i]
#    h = results["height"][i]
#    # extract the OCR text itself along with the confidence of the
#    # text localization
#    text = results["text"][i]
###    conf = int(float(results["conf"][i]))

#   # Open the file in append & read mode ('a+')
#    with open("TextRead.txt", "a+") as file_object:
#        # Move read cursor to the start of file.
#        file_object.seek(0)
#        # If file is not empty then append '\n'
#        data = file_object.read(100)
#        if len(data) > 0:
#            file_object.write("\n")
#        # Append text at the end of file
#        file_object.write(text)

# filter out weak confidence text localizations
###    if conf > min_conf:
#        # display the confidence and text to our terminal
#        print("Confidence: {}".format(conf))
#        print("Text: {}".format(text))
#        print("")
#        # strip out non-ASCII text so we can draw the text on the image
#        # using OpenCV, then draw a bounding box around the text along
#        # with the text itself
#        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
#        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
#                   1.2, (0, 0, 255), 3)
# show the output image
# cv2.imshow("Image", image)
# cv2.waitKey(0)


# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
#                help="path to input image to be OCR'd")
# ap.add_argument("-c", "--min-conf", type=int, default=0,
#                help="mininum confidence value to filter weak text detection")
# args = vars(ap.parse_args())


UPLOAD_FOLDER = 'C:\\Users\\mukre\\Postman\\files'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


class UploadImage(Resource):
    def post(self):
        parse = reqparse.RequestParser()
        parse.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files')
        args = parse.parse_args()
        image_file = args['file']
        image_file.save("your_file_name.jpg")

def current_milli_time():
    return round(time.time() * 1000)

@app.route("/get_image_text", methods=['POST', 'PUT'])
def baslangic_api():
    try:
        if request.method == 'POST':
            # check if the post request has the file part
            if 'image' not in request.files:
                return "-1 : No file part"

            file = request.files['image']

            if file.filename == '':
                return "-2 : No selected file"

            if allowed_file(file.filename):
                file_name = os.path.join(UPLOAD_FOLDER, file.filename)
                result_string = process_image_with_path(file_name)

                return result_string

            return "unsuccess"
    except Exception as e:
        return "error : " + e.__str__()

@app.route("/", methods=['GET'])
def helloo_api():
    return "Api anasayfa";

if __name__ == "__main__":
    app.run()
