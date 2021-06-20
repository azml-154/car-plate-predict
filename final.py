import cv2
import numpy as np
from tensorflow.keras.models import load_model

def yolo(frame, net, classes, layer_names, output_layers, colors):
    
    roi = frame
    img = cv2.resize(frame, None, fx=0.4, fy=0.4)
    
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.6:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)  -10
                h = int(detection[3] * height) -10
                # 좌표
                x = int(center_x - w / 2) 
                y = int(center_y - h / 2) 
                

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            
            if(label=='Plate'):
                area = w*h
                if(area >= 2500):
                    roi = img[y:y+h, x:x+w]
                
                
                    roi = cv2.resize(roi, dsize=(520, 110), interpolation=cv2.INTER_AREA)

                    #print(area)
                    #cv2.imshow('first_roi', roi)
                
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y-10), font, 3, color, 3)
            
    return img, roi








def affine(frame):
    # Read img
    ori_img = frame
    
    # Make Test Image
    #paper = cv2.imread('C:\paper.png')
    #_, paper = cv2.threshold(paper, 127, 255, cv2.THRESH_BINARY)
    
    #paper2 = cv2.imread('C:\paper.png')
    #_, paper2 = cv2.threshold(paper2, 127, 255, cv2.THRESH_BINARY)
    
    #paper3 = cv2.imread('C:\paper.png')
    #_, paper3 = cv2.threshold(paper3, 127, 255, cv2.THRESH_BINARY)

    # Show Original Image
    #cv2.imshow('ori', ori_img)

    # Color to Gray
    gray_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
    
    # Show Gray Image
    #cv2.imshow('gray',gray_img)

    # Binary(Otsu, Adaptive)
    _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)
    #binary_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 5)
    
    # Show Binary Image
    #cv2.imshow('binary', binary_img)
    #cv2.imshow('adaptive_binary',binary_img)
    
    # Gaussian Bluring
    gaussian_img = cv2.GaussianBlur(binary_img, (3, 3), 0)
    
    # Show Blur Image
    #cv2.imshow('gaussian', gaussian_img)
    
    # Canny Edge
    canny_img = cv2.Canny(gaussian_img,100,200)
    
    # Show CE Image
    #cv2.imshow('canny', canny_img)
    
    # Find Contours
    contours, hierahy = cv2.findContours(canny_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Array to Numpy
    contours = np.array(contours)
    
    # Draw Contours only
    #cv2.drawContours(paper2, contours, -1, (255,255,255), 1)
    
    # Erase Contours
    #cv2.drawContours(canny_img, contours, -1, (0,0,0), 1)

    # Show Result
    #cv2.imshow('erase_contour', canny_img)
    #cv2.imshow('paper2', paper2)
    # Information of Contours
    #print(contours)
    
    print('외곽선 개수:', len(contours))
    print('임의의 외곽선 좌표 개수:',len(contours[3]))
    #print('임의의 외곽선 좌표: ', contours[3])
   
    # Find Carplate's Contour
    contours_points = np.array([])
    for contour_index in range(0, len(contours)):
       contours_points = np.append(contours_points, len(contours[contour_index]))
    print('외곽선 좌표 개수:', contours_points)            
    
    car_plate = np.max(np.where(contours_points == np.max(contours_points)))
    print('번호판 좌표:', contours[car_plate])
    
    
    
    #------------------------------------------------------
    
    # Get Window Size
    y, x = ori_img.shape[:2]
    print('원본 이미지 사이즈:', ori_img.shape[:2])
    
    # Get Carplate's Points Location
    points = np. array(contours[car_plate])
    print(points.shape)
    
    # Find Minimum Distance
    points = np.reshape(points, (-1,2))
    print(points)
    #print(points.shape)
    
    dist_p1 = np.linalg.norm(points, ord=2, axis=1)
    #print(np.min(dist_p1))
    p1_index = np.argmin(dist_p1)
    #print(p1_index)
    p1 = points[p1_index]
    print(p1)
    
    dist_p2 = np.linalg.norm(points-(0, y), ord=2, axis=1)
    p2_index = np.argmin(dist_p2)
    p2 = points[p2_index]
    print(p2)
    
    dist_p3 = np.linalg.norm(points-(x, 0), ord=2, axis=1)
    p3_index = np.argmin(dist_p3)
    p3 = points[p3_index]
    print(p3)
    
    dist_p4 = np.linalg.norm(points-(x, y), ord=2, axis=1)
    p4_index = np.argmin(dist_p4)
    p4 = points[p4_index]
    print(p4)
    
    # Marking Points
    #cv2.line(paper, p1, p1, (255,0,0), 3)
    #cv2.line(paper, p2, p2, (255,0,0), 3)
    #cv2.line(paper, p3, p3, (255,0,0), 3)
    #cv2.line(paper, p4, p4, (255,0,0), 3)
    
    # Show Carplate's Contour
    #cv2.drawContours(paper, contours[car_plate], -1, (255,255,255), 1)
    #cv2.imshow('paper', paper)
    
    # Numpy to Array
    corner_points = np.float32([list(p1),list(p2),list(p3),list(p4)])
    resizing = np.float32([[0, 0], [0,110], [520,0], [520,110]])
    
    temp = cv2.getPerspectiveTransform(corner_points, resizing)
    affine = cv2.warpPerspective(ori_img, temp, (520,110))
    
    cv2.imshow('affine', affine)
    
    return affine
    
    
    
    
    
    
    
    
    
    
def find_string(roi):
    #grayscale transform
    ori_roi = roi
    ori_roi = cv2.resize(ori_roi, (520, 110), interpolation=cv2.INTER_NEAREST)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.GaussianBlur(roi, (3, 3), 0)
    #cv2.imshow('gray_roi', roi)
    
    
    
    #가우시안 블러링은 이미 원본 이미지에 수행됨 바로 이진화
    _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    roi = cv2.erode(roi, kernel)
    
    #cv2.imshow('first erode', roi)
    
    
    
    roi = cv2.bitwise_not(roi, 1)                                   #글씨를 흰색으로 변환
    roi = cv2.resize(roi, (520, 110), interpolation=cv2.INTER_NEAREST)

    #cv2.imshow('bitwise_not_binary', roi)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)
    
    roi = cv2.erode(roi, kernel)
    roi = cv2.GaussianBlur(roi, (3, 3), 0)
    
    #cv2.imshow('gray_roi', roi)

    
    #cv2.imshow('morphology', roi)

    

    contours, hierachy= cv2.findContours(roi.copy(), 
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = [cv2.boundingRect(each) for each in contours]
    #print(rects)
    
    tmp = [w*h for (x,y,w,h) in rects]
    tmp.sort()
    #print(tmp)
    
    
    rects = [(x,y,w,h) for (x,y,w,h) in rects if ((w*h>1000)and(w*h<13000) and (w/h<1.2))]
    #print(rects)
    
    for rect in rects:
    # Draw the rectangles
        cv2.rectangle(ori_roi, (rect[0], rect[1]), 
                  (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
        #print(rect)

    cv2.imshow('box',ori_roi)
    #cv2.imshow('roi', roi)


    #sort rects
    rect_list = []
    temp = []
    for i in rects:
        temp.append(i[0])
    temp.sort()
    
    for j in range(len(rects)):
        for i in rects:
            if(temp[j] == i[0]):
                rect_list.append(i)
                
    #print(rect_list)
            
    
    return roi, rect_list
    






def predict(roi, rect_list):
    num = []
    model = load_model('OTSU_Gausian.h5')
    
    for i in rect_list:
        
        
        x, y, w, h = i
        #print(w, h)
        pred_img = roi[i[1]:i[1] + i[3], i[0]:i[0] + i[2]]
        

    
        pred_img = cv2.resize(pred_img, (int(w)+10, int(h*3)), interpolation=cv2.INTER_AREA)
        h = int(pred_img.shape[0]/2)
        w = int(pred_img.shape[1]/2)
        
    
        mask = np.zeros([200, 200])
        mask[100 - h:100 - h +pred_img.shape[0], 100 - w:100 - w + pred_img.shape[1]] = pred_img
        
        mask = cv2.GaussianBlur(mask, (3, 3), 1)
        
        
        #cv2.imshow('num', mask)
        #cv2.waitKey(0)
        
        
        pred_img = cv2.resize(mask, (28,28), interpolation=cv2.INTER_AREA)
        
        

        
        pred_img = pred_img/255.
        input_img = pred_img.reshape(1,28,28,1)
        

        
        
        pred = model.predict(input_img)
        number = np.argmax(pred)
        num.append(str(number))
        num_str = "".join(num) 
    
    
    a = num_str[:2]
    b = num_str[3:]
    num_str = a + "-" + b
        
    return num_str




def final_plate(plate_list):
    for i in range(len(plate_list)):
        count=0
        for j in range(len(plate_list)):
            if(plate_list[i] == plate_list[j]):
                count+=1
                if(count > len(plate_list)/2):
                    #print('final : ',plate_list[i])
                    return(plate_list[i])
                    break





    
def camera_in():
    fps = 5 #꼭 홀수일것. 신뢰도를 높이기 위해서는 숫자를 키울 것
    plate_list = []


    
    
    
    
    cap = cv2.VideoCapture('car.mp4')
    
    
    net = cv2.dnn.readNet("yolov3_last.weights", "yolov3.cfg")
    classes = ['Car', 'Plate', 'car', 'plate', 'Car_plate', 'car_plate']
    #with open("coco.names", "r") as f:
    #    classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    while True:
        num_str= ''
        
        ret, frame = cap.read() #480, 640, 3
        frame = frame[1080-960:,:1280]

        frame, roi = yolo(frame, net, classes, layer_names, output_layers, colors)


        
        if(roi.shape == (110, 520, 3)):
            roi, rect_list = find_string(roi)
            
            if(len(rect_list) >= 7):
                #cv2.imshow('7', roi)
                num_str = predict(roi, rect_list)
                plate_list.append(num_str)
                

        
        #print(num_str)
    
        
        

        
        if (len(plate_list)%fps == 0):
            plate = final_plate(plate_list)
            if (plate !=None):
                print('번호판 출력값 :',plate)


            plate_list = []
        
        
        #if(num_str != ''):
        #    print(num_str)
            
        





        
        cv2.imshow('frame', frame)

        if cv2.waitKey(10) > 0:
            break

    cv2.destroyAllWindows()
    
camera_in()