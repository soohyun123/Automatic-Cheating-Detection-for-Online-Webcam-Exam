from argparse import ArgumentParser
import cvlib as cv
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer
import checkpoint
from yolo import YOLO
import marknet

##############################
#######   User Input   #######
##############################

parser = ArgumentParser()
parser.add_argument("--video", type=str, default=None,
                    help="Video file to be processed.")
parser.add_argument("--cam", type=int, default=None,
                    help="The webcam index.")
args = parser.parse_args()


##############################
#######    Constants   #######
##############################

outfile = "./video_out/example.avi"

name = 'train_1_5CNN'

CNN_INPUT_SIZE = 64

# Device configuration
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')


##############################
#######  Main Program  #######
##############################

def main():

    # Video source from webcam or video file.
    video_src = args.cam if args.cam is not None else args.video
    if video_src is None:
        print("Video source is not assigned, default webcam will be used.")
        video_src = 0

    # open webcam or video file
    webcam = cv2.VideoCapture(video_src)
    
    # Frame size adjustment
    #if video_src == 0:
    #    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
     
    if not webcam.isOpened():
        print("Could not open the video source")
        exit()
    
    #webcam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    _, sample_frame = webcam.read()

    height, width = sample_frame.shape[:2]
    fps = webcam.get(cv2.CAP_PROP_FPS)

    # To save output videos
    #fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    #out = cv2.VideoWriter(outfile, fourcc, fps, (width, height))

    
    # Introduce hand detector
    yolo = YOLO("./cfg/cross-hands-tiny.cfg", "./cfg/cross-hands-tiny.weights", ["hand"])
    yolo.size = 416
    yolo.confidence = 0.1
    
    # Introduce pose estimator
    pose_estimator = PoseEstimator(img_size=(height, width))

    # Introduce scalar stabilizers for pose.
    pose_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1) for _ in range(6)]

    # Introduce mark detector
    ckpt_dir = './marknet_chekpoints/' + name
    ckpt = checkpoint.load_checkpoint(ckpt_dir, load_best=True)

    D = marknet.marknet1()
    D = D.to(device)
    D.load_state_dict(ckpt['D'])
    D.eval()

    torch.no_grad()

    tm = cv2.TickMeter()

    # loop through frames
    while webcam.isOpened():
     
        # read frame from webcam 
        status, frame = webcam.read()
     
        if not status:
            print("Could not read frame")
            exit()
     
        # apply face detection
        faces, confidence = cv.detect_face(frame, threshold=0.5)
     
        #print(faces)
        #print(confidence)

        # No faces
        if len(faces) == 0:
            cv2.putText(frame, "ALERT!! No student", (175, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        

        # More than two people
        if len(faces) >= 2:
            for idx, f in enumerate(faces):
        
                (startX, startY) = f[0], f[1]
                (endX, endY) = f[2], f[3]

                cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)


            cv2.putText(frame, "ALERT!! More than one student ", (67, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            

        if len(faces) == 1:

            ##############################
            ####### Hand Detection #######
            ##############################

            ## if no hands then show the alert
            width, height, inference_time, hands = yolo.inference(frame)

            if len(hands) == 0:
                cv2.putText(frame, "ALERT!! No hands", (175, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                for detection in hands:
                    id_, name_, confidence, x, y, w, h = detection
                    if confidence > 0.1:
                        cx = x + (w / 2)
                        cy = y + (h / 2)
                        color = (0, 255, 255)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        text = "%s (%s)" % (name_, round(confidence, 2))
                        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


            ##############################
            ###  Head Pose Estimation  ###
            ##############################

            facebox = faces[0]

            #(startX, startY) = facebox[0], facebox[1]
            #(endX, endY) = facebox[2], facebox[3]

            facebox_s = utils.get_square_box(facebox)

            if utils.box_in_image(facebox_s, frame) == False:
                cv2.putText(frame, "ALERT!! student is out of cam frame", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Real-time cheating detection", frame)
                continue

            face_img = frame[facebox_s[1]: facebox_s[3],
                             facebox_s[0]: facebox_s[2]]
            face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

            face_img = torch.tensor(face_img)
            face_img = face_img.type(torch.cuda.FloatTensor).to(device)
            face_img = torch.unsqueeze(face_img, 0)
            face_img = torch.unsqueeze(face_img, 0)

            tm.start()
            marks = D(face_img) #(N, 1, 64, 64) -> (N, 68, 2)
            ## marks should have (68,2) shape
            ## and each value of marks is in the normalized scale of [0,1]
            tm.stop()

            marks = marks.detach().cpu().numpy()
            marks = np.reshape(marks, (-1, 68, 2)).squeeze()

            # Convert the marks locations from local CNN to the frame image.
            marks *= (facebox_s[2] - facebox_s[0])
            marks[:, 0] += facebox_s[0]
            marks[:, 1] += facebox_s[1]


            pose = pose_estimator.solve_pose_by_68_points(marks)

            # Stabilize the pose.
            steady_pose = []
            pose_np = np.array(pose).flatten()
            for value, ps_stb in zip(pose_np, pose_stabilizers):
                ps_stb.update([value])
                steady_pose.append(ps_stb.state[0])
            steady_pose = np.reshape(steady_pose, (-1, 3))

            rotationVector, translationVector = steady_pose[0], steady_pose[1]
            #rotationVector, translationVector = pose[0], pose[1]



            ##############################
            ######  Face Grahpics  #######
            ##############################

            # Draw tilted facebox
            pose_estimator.draw_annotation_box(
                frame, rotationVector, translationVector, color=(128, 255, 128))

            # Draw gaze direction line from the tip of the nose
            noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
            noseEndPoint2D, jacobian = cv2.projectPoints(
                noseEndPoints3D, rotationVector, translationVector, pose_estimator.camera_matrix, pose_estimator.dist_coeefs)

            p1 = (int(marks[30, 0]), int(marks[30, 1])) # marks[30] is the postion of the tip of the nose
            p2 = (int(noseEndPoint2D[0, 0, 0]), int(noseEndPoint2D[0, 0, 1]))
            cv2.line(frame, p1, p2, (110, 220, 0),
                     thickness=2, lineType=cv2.LINE_AA)

            # calculating euler angles
            rmat, jac = cv2.Rodrigues(rotationVector)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            #print('*' * 80)
            # print(f"Qx:{Qx}\tQy:{Qy}\tQz:{Qz}\t")
            x = np.arctan2(Qx[2][1], Qx[2][2])
            y = np.arctan2(-Qy[2][0], np.sqrt((Qy[2][1] * Qy[2][1] ) + (Qy[2][2] * Qy[2][2])))
            z = np.arctan2(Qz[0][0], Qz[1][0])

            #print("ThetaX: ", x)
            #print("ThetaY: ", y)
            #print("ThetaZ: ", z)
            #print('*' * 80)

            if angles[1] < -15:
                GAZE = "Looking: Left"
                cv2.putText(frame, "ALERT!! " + GAZE, (130, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif angles[1] > 15:
                GAZE = "Looking: Right"
                cv2.putText(frame, "ALERT!! " + GAZE, (130, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                GAZE = "Looking: Forward"

            cv2.putText(frame, GAZE, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)

     
        # display output
        cv2.imshow("Real-time cheating detection", frame)
        #out.write(frame)
     
        # press "Q" to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    # release resources
    webcam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()



# Reference
# https://github.com/yinguobing/head-pose-estimation
# https://github.com/by-sabbir/HeadPoseEstimation