#import the neccessary packages
import argparse
import requests
import time 
import os
import sys 


#construct the argument parser
ap=argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,help="path to output directory of images")
ap.add_argument("-n", "--num-images", type=int,default=500, help="# of images to download")
args = vars(ap.parse_args())

# initialize the URL that contains the captcha images that we will
# be downloading along with the total number of images downloaded
# thus far. Everytime you refresh this link a new captcha is generated
url = "https://www.e-zpassny.com/vector/jcaptcha.do"
total = 0

#You need the headers otherwise it won't work
headers = {
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36'}
# loop over the number of images to download
for i in range(0, args["num_images"]):
    try:
        # try to grab a new captcha image
        r = requests.get(url, headers=headers, timeout=5)

        #create the filenames for the images we are downloading.
        #NOTE: string method zfill() pads string on the left with zeros to fill width
        p = os.path.sep.join([args["output"], "{}.jpg".format(str(total).zfill(5))])
        f = open(p, "wb")
        f.write(r.content)
        f.close()

        # update the counter
        print("[INFO] downloaded: {}".format(p))
        total += 1
    except:
        #This will catch all exceptions
        print("[INFO] error downloading image...")
        e = sys.exc_info()[0]
        print(e)
    # insert a small sleep to be courteous to the server
    #time.sleep(0.1)