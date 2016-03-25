import numpy as np
import cv2

def capture_video_simple():
	cap = cv2.VideoCapture(0)
	if cap.isOpened():
		while(True):
		  # Capture frame-by-frame
		  ret, frame = cap.read()

		  # Our operations on the frame come here
		  # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		  # cv2.imshow('frame', gray)

		  # Display the resulting frame
		  cv2.imshow('frame', frame)
		  if cv2.waitKey(100) & 0xFF == ord('q'):
		    break

		# When everything done, release the capture
		cap.release()
		cv2.destroyAllWindows()

if __name__ == '__main__':
	capture_video_simple()

