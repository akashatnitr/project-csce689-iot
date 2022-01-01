import os
from os import listdir
from os.path import isfile, join
from glob import glob

from PIL import Image


def resize(loc):
    basewidth = 800

    img = Image.open(loc)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    img.save(loc)


file_name = '/Users/akash.sahoo/Documents/samsung/2020/door_intelligence/face/rpi/dataset/faces_to_download.txt'
with open(file_name) as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content]
# content
num_images = 20
file_path = 'python /Users/akash.sahoo/Documents/samsung/2020/door_intelligence/face/rpi/dataset/downloader_images/bing_scraper.py'


def rename(face_folder, onlyfiles):
    for i in range(len(onlyfiles)):

        try:
            os.rename(face_folder + onlyfiles[i],
                      face_folder + str('%0.8d' % i) + '.jpg')
            resize(face_folder + str('%0.8d' % i) + '.jpg')
        except Exception as e:
            print(e)
            try:
                os.remove(face_folder + str('%0.8d' % i) + '.jpg')
            except Exception as e:
                print(e)


#         print ('%0.8d' % i)
#         print ('%0.8d' % i)

for i in range(len(content)):
    args = "--search " + "'" + content[
        i] + "' --limit 20 --download --chromedrive /usr/local/bin/chromedriver"

    os.system(file_path + ' ' + args)

main_folder = '/Users/akash.sahoo/Documents/samsung/2020/door_intelligence/face/rpi/dataset/images/'
all_folder = glob(main_folder + "*/")

for face_folder in all_folder:
    # face_folder = '/Users/akash.sahoo/Documents/samsung/2020/door_intelligence/face/rpi/tmp_del/images/Kaley_Cuoco/'
    onlyfiles = [
        f for f in listdir(face_folder) if isfile(join(face_folder, f))
    ]
    rename(face_folder, onlyfiles)
# all_folder
# args = "--search 'Kaley_Cuoco' --limit 50 --download --chromedrive /usr/local/bin/chromedriver"
