import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width of the camera feed
cap.set(4, 720)   # Height of the camera feed
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)
colorR = (255, 0, 255)

class DragRect():
    def __init__(self, posCenter, size=[200, 200]):
        self.posCenter = posCenter
        self.size = size
        self.dragging = False  # Flag to check if dragging is active

    def update(self, cursor):
        if self.dragging:
            self.posCenter = cursor

rectList = [DragRect([x * 250 + 150, 150]) for x in range(5)]

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, draw=True, flipType=True)

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        if len(lmList) >= 12:
            index_finger = lmList[8][:2]  # Index finger tip coordinates
            thumb = lmList[4][:2]         # Thumb tip coordinates

            # Calculate distance between index finger and thumb
            length = cv2.norm(np.array(index_finger), np.array(thumb))
            cv2.putText(img, f"Length: {length}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if length < 50:  # Assuming 50 as the pinch threshold
                cursor = index_finger
                for rect in rectList:
                    if not rect.dragging:
                        if rect.posCenter[0] - rect.size[0] // 2 < cursor[0] < rect.posCenter[0] + rect.size[0] // 2 and rect.posCenter[1] - rect.size[1] // 2 < cursor[1] < rect.posCenter[1] + rect.size[1] // 2:
                            rect.dragging = True
                    if rect.dragging:
                        rect.update(cursor)
            else:
                # Release all dragging
                for rect in rectList:
                    rect.dragging = False

    # Drawing rectangles
    imgNew = np.zeros_like(img, np.uint8)
    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        cv2.rectangle(imgNew, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), colorR, cv2.FILLED)
        cvzone.cornerRect(imgNew, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)

    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    cv2.imshow("Image", out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
