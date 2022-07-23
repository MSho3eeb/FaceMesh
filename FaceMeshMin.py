import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture('Video/1.mp4')
cTime = 0
pTime = 0
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
mpPose = mp.solutions.pose
pose = mpPose.Pose()
facemesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=4, circle_radius=1)
mpHands = mp.solutions.hands
hands = mpHands.Hands()


while True:
    success, img = cap.read()
    success, img2 = cap2.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = facemesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLm in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img2, faceLm, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)

    # results2 = pose.process(imgRGB)
    # if results2.pose_landmarks:
    #     mpDraw.draw_landmarks(img2, results2.pose_landmarks, mpPose.POSE_CONNECTIONS)
    #     for id, lm in enumerate(results2.pose_landmarks.landmark):
    #         h, w, c = img2.shape
    #         cx = int(lm.x * w)
    #         cy = int(lm.y * h)
    #         #print(id, cx, cy)
    #         cv2.circle(img2, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    results3 = hands.process(imgRGB)
    if results3.multi_hand_landmarks:
        for handLms in results3.multi_hand_landmarks:
            mpDraw.draw_landmarks(img2, handLms, mpHands.HAND_CONNECTIONS)
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img2.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img2, (cx, cy), 10, (255, 0, 255), cv2.FILLED)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img2, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 4)
    cv2.imshow("OUTPUT", img2)
    cv2.waitKey(1)
