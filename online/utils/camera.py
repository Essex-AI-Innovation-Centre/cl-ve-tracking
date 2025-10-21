import numpy as np
import subprocess
import cv2

def find_camera_index(camera, source):
    p = subprocess.Popen("v4l2-ctl --list-devices", stdout=subprocess.PIPE, shell=True)
    str_devices_response = str(p.communicate())
    usb_index = str_devices_response.find(camera)
    if usb_index == -1:
        raise Exception("Camera not found!")
    str_devices = str_devices_response[usb_index:]
    prefix = "/dev/video"
    usb_index = str_devices.find(prefix)
    str_channels = str(str_devices[usb_index:])
    end_index = str_channels.find("\\n\\n")
    str_channels = str_channels[:end_index].split('\\n\\t')
    channels = [int(str_ch[len(prefix):]) for str_ch in str_channels if str_ch.find(prefix) != -1]
    index = channels.index(source)
    return index

def extract_calibration_matrix(calib_path, auto_bf=True):
    fs = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)
    M = np.array(fs.getNode('M1').mat(), dtype=np.float64)
    fx = M[0][0]
    fy = M[1][1]
    cx = M[0][2]
    cy = M[1][2]
    if auto_bf:
        bf = 'auto'
    else:
        T = np.array(fs.getNode('T').mat(), dtype=np.float64)
        b = np.linalg.norm(T)
        bf = b * fx
    return fx, fy, cx, cy, bf
