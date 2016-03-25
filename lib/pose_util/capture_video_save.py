import os
import sys
import cv2
import numpy as np

def mkdirs(filepath):
  if not os.path.isdir(filepath):
    os.makedirs(filepath)

def img_prefix(idx, prefix_n=6, prefix_str="0"):
  idx = str(idx)
  left_n = prefix_n - len(idx)
  rstr = prefix_str*left_n + idx 
  return rstr

def rotate(image, angle, center = None, scale = 1.0):
  (h, w) = image.shape[:2]
  if center is None:
      center = (w / 2, h / 2)
  # Perform the rotation
  M = cv2.getRotationMatrix2D(center, angle, scale)
  rotated = cv2.warpAffine(image, M, (w, h))

  return rotated

# capture video and save
def capture_video_save(out_dire, out_file, is_video=False, fps=18, vid_ext=".avi", img_ext=".jpg"):
  '''
  See here for more details..
    http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
    http://blog.csdn.net/poi7777/article/details/39736273
  '''
  # open camera
  cap = cv2.VideoCapture(0)
  cap.set(cv2.cv.CV_CAP_PROP_FPS, fps)

  if is_video:
    # output path
    out_path = out_dire + out_file + vid_ext

    # Define the codec and create VideoWriter object
    # fourcc = cv2.cv.CV_FOURCC(*'X264')
    fourcc = cv2.cv.CV_FOURCC('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(out_path, fourcc, fps, (720,480))

    # Start
    while(cap.isOpened()):
      ret, frame = cap.read()
      if ret==True:
        frame = cv2.flip(frame,0)
        # flipp frame
        frame = rotate(frame, 180)
        out.write(frame)
        print out_path
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
      else:
        break
  else:
    idx = 1
    while(cap.isOpened()):
      # Capture frame-by-frame
      ret, frame = cap.read()

      # # out operations on the frame come here
      # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      # # Display the resulting frame
      # cv2.imshow('frame',gray)

      cv2.imshow("frame", frame)
      out_path = out_dire + out_file + img_prefix(idx) + img_ext
      cv2.imwrite(out_path, frame)
      idx += 1
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
  # Release everything if job is finished
  cap.release()
  out.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  is_video = False
  out_file = "mude.16.02.20."
  out_dire = "/home/ddk/malong/dataset/person.torso/demo/"
  if is_video:
    out_dire = out_dire + "video/"
  else:
    out_dire = out_dire + "images/"
  mkdirs(out_dire)

  capture_video_save(out_dire, out_file, is_video)