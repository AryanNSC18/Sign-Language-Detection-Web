import os
import cv2

my_data = "path to save your own data"
no_of_classes = 26
dataset_size = 100 # Number of images per class

cap = cv2.VideoCapture(0)
for iter in range(no_of_classes):
    if not os.path.exists(os.path.join(my_data,str(iter))):
        os.makedirs(os.path.join(my_data,str(iter)))

    print('Collecting data for class {}'.format(iter))

    flag = False
    while True:
        res,frame = cap.read()
        cv2.putText(frame,'To start collection press "S"',(100,150),cv2.FONT_HERSHEY_SIMPLEX,1.3,(255,0,0),3)
        cv2.imshow('Cap_window',frame)
        if cv2.waitKey(25)==ord('s'):
            break
        counter = 0
        while counter<dataset_size:
            res,frame = cap.read()
            cv2.imshow('Frame_set', frame)
            cv2.waitKey(25)
            cv2.imwrite(os.path.join(my_data, str(iter), '{}.jpg'.format(counter)), frame)

            counter += 1

cap.release()
cv2.destroyAllWindows()