import cv2
import face_recognition
import pickle
import os
from django.conf import settings
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db, storage
from django.conf import settings

# cred = credentials.Certificate("D:/MyUniversity/HK222/DADN/AIModule/face_recognition_app/recognition/Database/serviceAccountKey.json")

# firebase_admin.initialize_app(cred, {
#     'databaseURL': "https://face-recognition-92928-default-rtdb.asia-southeast1.firebasedatabase.app/",
#     'storageBucket': "face-recognition-92928.appspot.com"
# })


class Trainer:
    folderProfile = settings.PROFILES_PATH
    profileList = []
    memberList = []
    nameList = []
    encodeList = []

    def __init__(self):
        print(self.folderProfile)
        pathList = os.listdir(self.folderProfile)

        for path in pathList:
            self.profileList.append(cv2.imread(os.path.join(self.folderProfile, path)))  
            Id, name = os.path.splitext(path)[0].split('-')

            self.memberList.append((Id, name))
            # fileName = f'{self.folderProfile}/{path}'
            # bucket = storage.bucket()
            # blob = bucket.blob(fileName)
            # blob.upload_from_filename(fileName)

        self.__findEncodings()
        encodeWithId = [self.encodeList, self.memberList]

        self.__store(encodeWithId)


    def __findEncodings(self):
        for img in self.profileList:
            # rgb_img = img[:, :, ::-1]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            self.encodeList.append(encode)

    def __store(self, encodeWithId):
        file = open(f"{settings.PATH_ENCODING}/EncodeFile.p", "wb")
        pickle.dump(encodeWithId, file)
        file.close()
        pass
    

if __name__ == '__main__':
    trainer = Trainer()