import os
import matlab.engine
import sys


# start matlab by python script.
eng = matlab.engine.start_matlab()

# Define a folder that spherical images going to be saved.
OmniSaveDir = '../img_files/{}/'.format(sys.argv[1])
if not os.path.exists(OmniSaveDir):
    os.makedirs(OmniSaveDir)

# Check the path where images from streetview are saved.
ImgSaveDir = '../img_files/raw_{}/'.format(sys.argv[1])

if len(os.listdir(ImgSaveDir)) == len(os.listdir(OmniSaveDir)):
    print "{} images saving completed.".format(len(os.listdir(OmniSaveDir)))
    quit()

for dir in os.listdir(ImgSaveDir):
    if dir+'.jpg' not in os.listdir(OmniSaveDir):
        targetdir = ImgSaveDir + dir + '/StitchedImgs/' + dir + '.jpg'
        eng.CartesianToOmni(targetdir, sys.argv[1], nargout=0)
        print "{}/{} images saved.".format(len(os.listdir(OmniSaveDir)), len(os.listdir(ImgSaveDir)))


"""
# ricoh theta version; tested in 21 jul 2018.

for dir in os.listdir(ImgSaveDir):
    if dir+'.jpg' not in os.listdir(OmniSaveDir):
        targetdir = ImgSaveDir + dir
        #targetdir = ImgSaveDir + dir + '/StitchedImgs/' + dir + '.jpg'
        eng.CartesianToOmni(targetdir, sys.argv[1], nargout=0)
        print "{}/{} images saved.".format(len(os.listdir(OmniSaveDir)), len(os.listdir(ImgSaveDir)))
"""


"""
print os.listdir(ImgSaveDir)
for dir in os.listdir(ImgSaveDir):
    targetdir = ImgSaveDir + dir + '/StitchedImgs/' + dir +'.jpg'
    eng.CartesianToOmni(targetdir, nargout = 0)
"""
