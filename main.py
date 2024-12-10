import cv2
import numpy as np
import math
START_POINT = 150
END_POINT = 100
CLASSES = ["res", "cap", "ind", "dio", "led", "ic","ot" ]
# thứ tự class đã training
elec_classes = [0, 1, 2, 36, 4, 5, 6]
#load file config vaf file trọng số
YOLOV3_CFG = 'yolov3.cfg.txt'
YOLOV3_WEIGHT = 'yolov3_last.weights'
# ngưỡng tin cậy
CONFIDENCE_SETTING = 0.4
# kích thước video để nhận dạng
YOLOV3_WIDTH = 416
YOLOV3_HEIGHT = 416
#khoảng cách cập nhật 
#80
MAX_DISTANCE = 80
def get_output_layers(net):
    try:
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers
    except:
        print("Can't get output layers")
        return None
def detections_yolo3(net, image, confidence_setting, yolo_w, yolo_h, frame_w, frame_h, classes=None):
    # Thay đổi kích thước ảnh đầu vào về kích thước yêu cầu của mô hình YOLO
    img = cv2.resize(image, (yolo_w, yolo_h))
    # Tạo blob từ ảnh đã thay đổi kích thước, chuẩn bị dữ liệu đầu vào cho mô hình YOLO
    blob = cv2.dnn.blobFromImage(img, 0.00392, (yolo_w, yolo_h), swapRB=True, crop=False)
    net.setInput(blob)
    # Chạy mô hình để lấy các lớp đầu ra của YOLO
    layer_output = net.forward(get_output_layers(net))
    # Khởi tạo danh sách để lưu các thông tin phát hiện
    boxes = []
    class_ids = []
    confidences = []
    # Lặp qua các đầu ra của mỗi lớp trong YOLO
    for out in layer_output:
        # Lặp qua từng phát hiện trong các lớp đầu ra
        for detection in out:
            # Lấy các giá trị xác suất cho các lớp (từ phần tử thứ 6 trở đi)
            scores = detection[5:]
            # Tìm ID của lớp có xác suất cao nhất
            class_id = np.argmax(scores)
            # Lấy giá trị xác suất của lớp được dự đoán cao nhất
            confidence = scores[class_id]

            # Kiểm tra nếu xác suất cao hơn ngưỡng yêu cầu và đối tượng thuộc lớp xe cộ
            if confidence > confidence_setting and class_id in elec_classes:
                # In ra thông tin đối tượng được phát hiện cùng độ tin cậy
                print(f"Phát hiện {classes[class_id]} với mức tin cậy {confidence * 100:.2f}%")

                # Tính toán tọa độ trung tâm và kích thước của hộp bao quanh (bounding box)
                center_x = int(detection[0] * frame_w)  # Tọa độ X của tâm
                center_y = int(detection[1] * frame_h)  # Tọa độ Y của tâm
                w = int(detection[2] * frame_w)  # Chiều rộng của hộp
                h = int(detection[3] * frame_h)  # Chiều cao của hộp

                # Tính toán tọa độ góc trên bên trái của hộp bao quanh
                x = center_x - w / 2
                y = center_y - h / 2

                # Lưu thông tin phát hiện vào các danh sách
                class_ids.append(class_id)  # Lưu ID lớp phát hiện
                confidences.append(float(confidence))  # Lưu độ tin cậy của phát hiện
                boxes.append([int(x), int(y), int(w), int(h)])  # Lưu hộp bao quanh

    # Trả về danh sách hộp bao quanh, ID lớp và độ tin cậy của từng phát hiện
    return boxes, class_ids, confidences

def draw_prediction(classes, colors, img, class_id, confidence, x, y, width, height):
    try:
        label = str(classes[class_id])
        color = colors[class_id]
        center_x = int(x + width / 2.0)
        center_y = int(y + height / 2.0)
        x = int(x)
        y = int(y)
        width = int(width)
        height = int(height)
        cv2.rectangle(img, (x, y), (x + width, y + height), color, 1)
        cv2.circle(img, (center_x, center_y), 2, (0, 255, 0), -1)
        cv2.putText(img, label + ": {:0.2f}%".format(confidence * 100), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    except (Exception, cv2.error) as e:
        print("Không thể nhận dạng id {}: {}".format(class_id, e))
def check_location(box_y, box_height, height):
    center_y = int(box_y + box_height / 2.0)
    if center_y > height - END_POINT:
        return True
    else:
        return False
def check_start_line(box_y, box_height):
    center_y = int(box_y + box_height / 2.0)
    if center_y > START_POINT:
        return True
    else:
        return False
def counting_vehicle(video_input, video_output, skip_frame=1):
    colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    # Load mô hình yolo
    net = cv2.dnn.readNetFromDarknet(YOLOV3_CFG, YOLOV3_WEIGHT) #import mô hình YOlOv3
    # Đọc khung hình đầu tiên
    cap = cv2.VideoCapture(video_input)
    ret_val, frame = cap.read()
    #Lấy kích thước của khung hình
    width = frame.shape[1]
    height = frame.shape[0]
    # Định dạng video ra
    video_format = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_output, video_format, 25, (width, height)) # định dạng vid output
    # Khởi tạo số lkien ban đầu
    list_object = []
    number_frame = 0
    number_total = 0
    number_res = 0
    number_cap = 0
    number_ind = 0
    number_dio = 0
    number_led = 0
    number_ic = 0
    number_ot = 0
    while cap.isOpened():
        number_frame += 1
        ret_val, frame = cap.read()
        if frame is None:
            break
        tmp_list_object = list_object
        list_object = []
        for obj in tmp_list_object: # cập nhật vị trí của nó tại frame hiện tại
            tracker = obj['tracker']
            class_id = obj['id']
            confidence = obj['confidence']
            check, box = tracker.update(frame)
            if check:
                box_x, box_y, box_width, box_height = box
                draw_prediction(CLASSES, colors, frame, class_id, confidence, # cập nhật thành công thi vẽ vị trí của nó
                                box_x, box_y, box_width, box_height)

                obj['tracker'] = tracker
                obj['box'] = box
                if check_location(box_y, box_height, height): # nếu check vượt qua endline thì +1
                    number_total += 1
                    if class_id == 0:
                        number_res +=1
                    elif class_id == 1:
                        number_cap +=1
                    elif class_id == 2:
                        number_ind +=1
                    elif class_id == 3:
                        number_dio +=1
                    elif class_id == 4:
                        number_led +=1
                    elif class_id == 5:
                        number_ic +=1
                    else:
                        number_ot +=1
                else:
                    list_object.append(obj) # chưa qua thì đưa trở lại list obj
        #detect những obj mới xuất hiện
        if number_frame % skip_frame == 0:
            boxes, class_ids, confidences = detections_yolo3(net, frame, CONFIDENCE_SETTING, YOLOV3_WIDTH,
                                                             YOLOV3_HEIGHT, width, height, classes=CLASSES)
            for idx, box in enumerate(boxes):
                box_x, box_y, box_width, box_height = box
                if not check_location(box_y, box_height, height):
                    box_center_x = int(box_x + box_width / 2.0)
                    box_center_y = int(box_y + box_height / 2.0)
                    check_new_object = True
                    for tracker in list_object:
                        current_box_x, current_box_y, current_box_width, current_box_height = tracker['box']
                        current_box_center_x = int(current_box_x + current_box_width / 2.0)
                        current_box_center_y = int(current_box_y + current_box_height / 2.0)
                        distance = math.sqrt((box_center_x - current_box_center_x) ** 2 +
                                             (box_center_y - current_box_center_y) ** 2)
                        if distance < MAX_DISTANCE:
                            check_new_object = False
                            break
                    if check_new_object and check_start_line(box_y, box_height):
                        if hasattr(cv2, 'legacy'):
                            new_tracker = cv2.legacy.TrackerKCF_create()
                        elif hasattr(cv2, 'TrackerKCF_create'):
                            new_tracker = cv2.TrackerKCF_create()
                        else:
                            print("Tracker not available")
                            continue

                        new_tracker.init(frame, tuple(box))
                        new_object = {
                            'id': class_ids[idx],
                            'tracker': new_tracker,
                            'confidence': confidences[idx],
                            'box': box
                        }
                        list_object.append(new_object)
                        draw_prediction(CLASSES, colors, frame, new_object['id'], new_object['confidence'],
                                        box_x, box_y, box_width, box_height)
        cv2.putText(frame, "Total: {:d}".format(number_total), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 0, 255), 2)
        cv2.putText(frame, "Res: {:d}".format(number_res), (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 0, 0), 2)
        cv2.putText(frame, "Cap: {:d}".format(number_cap), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 255, 0), 2)
        cv2.putText(frame, "Ind: {:d}".format(number_ind), (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (80, 127, 255), 2)
        cv2.putText(frame, "Dio: {:d}".format(number_dio), (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (242, 7, 191), 2)
        cv2.putText(frame, "Led: {:d}".format(number_led), (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (108, 0, 250), 2)
        cv2.putText(frame, "IC: {:d}".format(number_ic), (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 219, 22), 2)
        cv2.putText(frame, "Ot: {:d}".format(number_ot), (10, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (167, 250, 0), 2)
        cv2.line(frame, (0, START_POINT), (width, START_POINT), (0, 255, 0), 1)
        cv2.line(frame, (0, height - END_POINT), (width, height - END_POINT), (0, 0, 255), 2)
        cv2.imshow("Counting", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        out.write(frame)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    counting_vehicle('video3.mp4', 'video3.avi')
