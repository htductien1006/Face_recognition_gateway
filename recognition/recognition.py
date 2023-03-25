from imutils.video import VideoStream
from imutils.video import FPS
# import imutils
import cv2,os,urllib.request,pickle
import face_recognition
import numpy as np
from recognition.encoding import Trainer
import math
from django.conf import settings

class FaceDectect(object):
    faceLocations = []
    faceEncoding = []
    nameList = []
    knownFaceNames = []
    knownEncodedFace = []
    processCurrentFrame = True

    def __init__(self) -> None:
        self.encode_face()

    def __del__(self):
        cv2.destroyAllWindows()
    
    def encode_face(self) -> None:
        if not os.path.isfile(f'{settings.PATH_ENCODING}/EncodeFile.p'):
            print("encode file is empty")
            self.retrain()

        file = open(f'{settings.PATH_ENCODING}/EncodeFile.p', "rb")
        if file.tell() == 0:
            file.close()
            file = self.retrain()
        else:
            encodeWithIDs = pickle.load(file)
            if len(encodeWithIDs) < self.getNumOfMembers():
                file.close()
                file = self.retrain()
        encodeWithIDs = pickle.load(file)
        file.close()

        self.knownEncodedFace, knownMembers = encodeWithIDs
        for mem in knownMembers:
            self.knownFaceNames.append(mem[1])

    def retrain(self):
        print("Retraining....Please wait for minutes!")
        trainer = Trainer()
        return open(f'{settings.PATH_ENCODING}/EncodeFile.p', "rb")

    def getNumOfMembers(self):
        folderProfile = "Profiles"
        pathList = os.listdir(folderProfile)
        return len(pathList)
    
    def get_faceConfidence(self, faceDistance, faceMatchThreshold = 0.6):
        range = (1.0 - faceMatchThreshold)
        linearVal = (1.0 - faceDistance) / (range * 2.0)
        
        print("Face Distance:", faceDistance)
        if faceDistance > faceMatchThreshold:
            return str(round(linearVal * 100, 2)) + '%'
        else:
            value = (linearVal + ((1.0 - linearVal) * math.pow((linearVal - 0.5) * 2, 0.2))) * 100
            return str(round(value, 2)) + '%'
    
    def get_frame(self):
        if len(self.nameList) == 20:
            self.nameList.clear()
        self.videoCap = cv2.VideoCapture(0)

        # while True:
        ret, frame = self.videoCap.read()

        if self.processCurrentFrame:
            #Scale current frame down to 0.25 of real size in order to reduce computation
            smallFrame = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
            rgbSmallFrame = cv2.cvtColor(smallFrame, cv2.COLOR_BGR2RGB)

            #find all faces in the current frame
            self.faceLocations = face_recognition.face_locations(rgbSmallFrame)
            self.faceEncoding = face_recognition.face_encodings(rgbSmallFrame, self.faceLocations)

            for encodedFace in self.faceEncoding:
                matches = face_recognition.compare_faces(self.knownEncodedFace, encodedFace)
                name = 'Unknown'
                confidence = '???'

                print("Matches: ", matches)

                faceDistances = face_recognition.face_distance(self.knownEncodedFace, encodedFace)
                print("Distances: ", faceDistances)

                matchedIdx = np.argmin(faceDistances)
                print("Matched Index:", matchedIdx)
                print("Match value:", matches[matchedIdx])
                if matches[matchedIdx]:
                    name = self.knownFaceNames[matchedIdx]
                    confidence = self.get_faceConfidence(faceDistances[matchedIdx])

                self.nameList.append(f'{name} ({confidence})')

        self.processCurrentFrame = not self.processCurrentFrame
        #Display annotations
        for (top, right, bottom, left), name in zip(self.faceLocations, self.nameList):
            top, right, bottom, left = top*4, right*4, bottom*4, left*4

            print(name)
            if name == 'Unknown (???)':
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            else:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 128), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 128), -1)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
