import cv2
import dlib
import imutils
import logging
import winsound
from scipy.spatial import distance
from imutils import face_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s [INFO] %(message)s")

class Drowsy:
    def _init_(self, cfg):
        self.th = cfg["th"]
        self.req = cfg["frames"]
        self.pred = cfg["pred"]
        self.cam = cfg["cam"]

        self.det = dlib.get_frontal_face_detector()
        self.shape = dlib.shape_predictor(self.pred)

        (self.ls, self.le) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rs, self.re) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        self.cnt = 0
        self.alert = False
        self.cap = cv2.VideoCapture(self.cam)

    def ear(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def beep(self):
        winsound.Beep(1000, 500)

    def step(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.det(gray, 0)

        for f in faces:
            shape = self.shape(gray, f)
            shape = face_utils.shape_to_np(shape)

            le = shape[self.ls:self.le]
            re = shape[self.rs:self.re]

            ear_val = (self.ear(le) + self.ear(re)) / 2.0

            cv2.drawContours(frame, [cv2.convexHull(le)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(re)], -1, (0, 255, 0), 1)

            if ear_val < self.th:
                self.cnt += 1
                if self.cnt >= self.req:
                    if not self.alert:
                        self.alert = True
                        logging.warning("Drowsy!")
                        self.beep()
                    cv2.putText(frame, "DROWSY!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                self.cnt = 0
                self.alert = False

        return frame

    def run(self):
        logging.info("Starting...")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logging.error("No frame.")
                break

            frame = imutils.resize(frame, width=450)
            frame = self.step(frame)
            cv2.imshow("Drowsy", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                logging.info("Stopped.")
                break

        self.cap.release()
        cv2.destroyAllWindows()


if _name_ == "_main_":
    cfg = {
        "th": 0.25,
        "frames": 20,
        "pred": "shape_predictor_68_face_landmarks.dat",
        "cam": "http://192.168.1.100:8080/video"
    }

    d = Drowsy(cfg)
    d.run()