source /home/pi/.virtualenvs/coral/bin/activate
cd /home/pi/code/tmp/coral-classification-detection
python detect_person.py -m ./mobilenet_ssd_v2/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite -l ./mobilenet_ssd_v2/coco_labels.txt
