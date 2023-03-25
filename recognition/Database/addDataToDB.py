import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("D:/MyUniversity/HK222/DADN/AIModule/face_recognition_app/recognition/Database/serviceAccountKey.json")

firebase_admin.initialize_app(cred, {
    'databaseURL': "https://face-recognition-92928-default-rtdb.asia-southeast1.firebasedatabase.app/"
})

ref = db.reference('Members')

data = {
    '1':
        {
            "name": "Critiano Ronaldo",
            "role": "Guest"
        },
    '2':
        {
            "name": "Elon Musk",
            "role": "Guest"
        }
}

for key, value in data.items():
    ref.child(key).set(value)