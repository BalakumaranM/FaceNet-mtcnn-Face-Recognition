from scipy.spatial.distance import cosine
import numpy as np
import cv2
import mtcnn
from keras.models import load_model
from utils import get_face, plt_show, get_encode, load_pickle, l2_normalizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from utils import *
from numpy import asarray
from numpy import expand_dims
from datetime import datetime

encoder_model = 'data/models/facenet_keras.h5'
people_dir = 'data/people'
encodings_path = 'data/encodings/encodings.pkl'
test_img_path = 'data/test/friends.jpg'
test_res_path = 'data/results/friends.jpg'

recognition_t = 0.3
required_size = (160, 160)

encoding_dict = load_pickle(encodings_path)
face_detector = mtcnn.MTCNN()
face_encoder = load_model(encoder_model)

in_encoder = Normalizer(norm='l2')

model = SVC(kernel='linear' , probability=True)
y = []
X = []
for key in encoding_dict.keys():
        y.append(key)

print(y)
for value in encoding_dict.values():
    value = l2_normalizer.transform(value.reshape(1, -1))[0]
    value = value.reshape(1, -1)
    value = value[0]
    X.append(value)

# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(y)
y = out_encoder.transform(y)
print(X)


model.fit(X,y)


def recognize(img,
              detector,
              encoder,
              encoding_dict,i,
              recognition_t=0.5,
              confidence_t=0.99,
              required_size=(160, 160), ):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)

    for res in results:
        if res['confidence'] < confidence_t:
            continue
        face, pt_1, pt_2 = get_face(img_rgb, res['box'])
        encode = get_encode(encoder, face, required_size)

        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        encode = encode.reshape(1, -1)
        name = 'unknown'

        yhat_class = model.predict(encode)
        print(yhat_class)
        class_index = yhat_class[0]
        yhat_prob = model.predict_proba(encode)
        print(yhat_prob)
        class_probability = yhat_prob[0, class_index] * 100
        predict_names = out_encoder.inverse_transform(yhat_class)

        distance = float("inf")
        for db_name, db_encode in encoding_dict.items():

            # r_score = accuracy_score(i, r)


            dist = cosine(db_encode, encode)

            if dist < recognition_t and dist < distance:
                name = db_name
                distance = dist



        if name == 'unknown':
            cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
            cv2.putText(img, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        else:
            cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)
            cv2.putText(img, predict_names[0] + f'_probability_{class_probability:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 200, 200), 2)
            markAttendance(predict_names[0])
    return img


# img = cv2.imread(test_img_path)
# plt_show(img)
def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList =[]
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in  line:
            now = datetime.now()
            dt_string = now.strftime("%H:%M:%S")
            f.writelines(f'\n{name},{dt_string}')


vc = cv2.VideoCapture(0)
i=-1
while vc.isOpened():
    ret, frame = vc.read()
    if ret :
        i+=1
    if not ret:
        print("no frame:(")
        break
    frame = recognize(frame, face_detector, face_encoder, encoding_dict,i)

    cv2.imshow('camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        codec = cv2.VideoWriter_fourcc()
        out = cv2.VideoWriter(FLAGS.output, codec, frame, (160, 160))
        break


#
# cv2.imwrite(test_res_path, img)
# plt_show(img)

print('2')


print('3')