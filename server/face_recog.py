import os
import sys
import dlib
import cv2
import openface
import pickle


import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from scipy import misc

def recog_face_openface(file_path, output_path, classifier):
    img = cv2.imread(file_path);

    face_detector = dlib.get_frontal_face_detector()

    predictor_model = os.path.join(os.getcwd(), "..","models","shape_predictor_68_face_landmarks.dat")

    detected_faces = face_detector(img, 1)
    
    
    face_pose_predictor = dlib.shape_predictor(predictor_model)
    face_aligner = openface.AlignDlib(predictor_model)

    faces = []


    for i, face_rect in enumerate(detected_faces):

        pose_landmarks = face_pose_predictor(img, face_rect)

        alignedFace = face_aligner.align(96, img, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

        net = openface.TorchNeuralNet()
        embeddings = net.forward(alignedFace)

        file = open('../models/'+classifier+'.p', 'rb')
        clf = pickle.load(file)
        file.close()


        prediction = clf.predict_proba([embeddings]).ravel()
        
        
        if(prediction[0] > prediction[1]):
            name = 'modi'.format(max(prediction))
        else:
            name = 'kejriwal'.format(max(prediction))
            
        if(max(prediction) < 0.77):
            name = "someone else"

        faces.append(name)

        cv2.rectangle(img,(face_rect.left(),face_rect.top()),(face_rect.right(),face_rect.bottom()),(0,255,0),2)

        cv2.putText(img, name, (face_rect.left(),face_rect.top()), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        cv2.putText(img, str(max(prediction)), (face_rect.left(),face_rect.bottom()), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)


    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_path, img)
    return len(detected_faces), faces


def recog_face_resnet(file_path, output_path):
    img = cv2.imread(file_path);

    face_detector = dlib.get_frontal_face_detector()

    predictor_model = os.path.join(os.getcwd(), "..","models","shape_predictor_68_face_landmarks.dat")

    detected_faces = face_detector(img, 1)
    
    
    face_pose_predictor = dlib.shape_predictor(predictor_model)
    face_aligner = openface.AlignDlib(predictor_model)

    faces = []
    for i, face_rect in enumerate(detected_faces):

        pose_landmarks = face_pose_predictor(img, face_rect)

        alignedFace = face_aligner.align(224, img, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        cv2.imwrite('static/aligned.jpg', alignedFace)
        embeddings = get_embeddings('static/aligned.jpg')

        file = open('../models/knn_resnet.p', 'rb')
        clf = pickle.load(file)
        file.close()


        prediction = clf.predict_proba(embeddings).ravel()
        
        pred = clf.predict(embeddings)

        print(pred)
        if(prediction[0] > prediction[1]):
            name = 'modi'.format(max(prediction))
        else:
            name = 'kejriwal'.format(max(prediction))
            
        if(max(prediction) < 0.80):
            name = "someone else"

        faces.append(name)

        cv2.rectangle(img,(face_rect.left(),face_rect.top()),(face_rect.right(),face_rect.bottom()),(0,255,0),2)

        cv2.putText(img, name, (face_rect.left(),face_rect.top()), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        cv2.putText(img, str(max(prediction)), (face_rect.left(),face_rect.bottom()), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)


    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_path, img)
    return len(detected_faces), faces

def get_embeddings(img):
    with tf.Graph().as_default():

        with tf.Session() as sess:

            load_model("/home/neelansh/precog_task/models/20170512-110547/20170512-110547.pb")

            img = [misc.imread(img)]

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")


            feed_dict = { images_placeholder:img, phase_train_placeholder:False }
            return sess.run(embeddings, feed_dict=feed_dict).reshape((1, 128*9))



def load_model(model):
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        with gfile.FastGFile(model_exp,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('error file not found')