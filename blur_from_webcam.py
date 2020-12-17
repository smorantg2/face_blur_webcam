import cv2
import face_alignment
import numpy as np

def anonymize_face_simple(image, factor=3.0):
	# automatically determine the size of the blurring kernel based
	# on the spatial dimensions of the input image
	(h, w) = image.shape[:2]
	kW = int(w / factor)
	kH = int(h / factor)
	# ensure the width of the kernel is odd
	if kW % 2 == 0:
		kW -= 1
	# ensure the height of the kernel is odd
	if kH % 2 == 0:
		kH -= 1
	# apply a Gaussian blur to the input image using our computed
	# kernel size
	return cv2.GaussianBlur(image, (kW, kH), 0)


cam = cv2.VideoCapture(0)

# sfd for SFD, dlib for Dlib and folder for existing bounding boxes. or blazeface
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, face_detector='blazeface', device = "cuda")

while True:
    ret, frame = cam.read()
    width = frame.shape[1]
    height = frame.shape[0]
    # frame = cv2.pyrDown(frame)
	
	
    preds = fa.get_landmarks(frame)
    if preds is None:
    	continue
    xs = [d[0] for d in preds[0]]
    ys = [d[1] for d in preds[0]]
    xmin = np.min(xs)
    xmax = np.max(xs)
    ymin = np.min(ys)
    ymax = np.max(ys)

    difx = np.abs((xmax - xmin)/3)
    dify = np.abs((ymax - ymin)/3)

    x0 = int(np.max([xmin - difx, 0]))
    x1 = int(np.min([xmax + difx, width]))
    y0 = int(np.max([ymin - dify*2, 0]))
    y1 = int(np.min([ymax + dify, height]))

    # cv2.rectangle(frame, (x0, y0), (x1, y1), (0,0,0))

    # extract the face ROI
    face = frame[y0:y1, x0:x1]
    face = anonymize_face_simple(face, factor=2.0)


    # store the blurred face in the output image
    frame[y0:y1, x0:x1] = face

    for num,detection in enumerate(preds[0]):

        cv2.circle(frame, tuple(detection), 1, (255,0,0), thickness=4)
        # cv2.putText(frame, str(num), tuple(detection), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0,0,255))




    cv2.imshow("Show", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
