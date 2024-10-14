from collections import deque
from collections import namedtuple
import threading
import cv2
import playsound
import torch
import csv
import numpy as np

from stats import Statistics

# ###############
# Global Vars
CLASSES = [0, 2, 15, 16]
CONF = 0.30
IOU = 0.50
WINDOW_SIZE = 20
DETECTIONS_TO_ALARM = WINDOW_SIZE // 2

G_KERNEL_SIZE = (3, 3)
SHARPEN_KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
# ###############

# Carica il modello YOLOv5 pre-addestrato
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

model.classes = CLASSES # lista di classi di oggetti che vuoi rilevare
model.conf = CONF  # soglia di confidenza minima per considerare un rilevamento valido
model.iou = IOU  # rimozione di rilevamenti sovrapposti per mantenere solo il migliore (NMS) --> ridurre il numero di rilevamenti multipli dello stesso oggetto, ma rischia di non riconoscere oggetti molto vicini
model.agnostic = True # le classi diverse vengono trattate allo stesso modo nel NMS -->  ridurre i falsi positivi in situazioni dove ci sono molti oggetti di classi diverse, ma simili tra loro
model.max_det = 20 # numero massimo di oggetti rilevati per immagine
model.multi_label = False #  il modello non può assegnare più classi a un singolo rilevamento
#model.imgsz = 640 # Ridimensionare a una risoluzione più alta può migliorare il rilevamento di piccoli oggetti --> Ridimensionare a una risoluzione più alta può migliorare il rilevamento di piccoli oggetti ->
model.augment = True # applica l'augmentazione dei dati durante l'inferenza, che può migliorare la robustezza delle previsioni



# Definisco le statistiche
human_statistics_enhanced = Statistics()
animal_statistics_enhanced = Statistics()
car_statistics_enhanced = Statistics()

human_statistics_original = Statistics()
animal_statistics_original = Statistics()
car_statistics_original = Statistics()

human_statistics_reduced = Statistics()
animal_statistics_reduced = Statistics()
car_statistics_reduced = Statistics()

human_statistics_en_red = Statistics()
animal_statistics_en_red = Statistics()
car_statistics_en_red = Statistics()



def detect(frame):
    # Esegui l'inferenza con il modello YOLOv5
    results = model(frame)
    # Converti i risultati in un DataFrame Pandas
    results_df = results.pandas().xyxy[0]
    return results_df


def draw_frame(frame, res, name, color):
    for _, row in res.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        confidence = row['confidence']
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{name} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame


def play_sound(stop_thread):
    while not stop_thread.is_set():
        playsound.playsound('sounds/alarm.wav')


# Evento per segnalare al thread di terminare l'esecuzione
stop_event = threading.Event()
# Thread per riprodurre il suono di allarme
sound_thread = threading.Thread(target=play_sound, args=(stop_event,))
# Struttura dati per rappresentare una entry nel dizionario delle detections
dict_entry = namedtuple('dict_entry', ['value', 'color'])
# Dizionario delle detections
res_dict = dict()


def detect_human(res, slwin: deque[bool]):
    if not hasattr(detect_human, "ALARM"):
        detect_human.ALARM = False  # Variabile "statica" C-like

    if res.empty:
        slwin.append(False)
    else:
        slwin.append(True)

    detections = sum(slwin)
    if detections >= DETECTIONS_TO_ALARM:
        # print("ALARM!")
        if not detect_human.ALARM:
            sound_thread.start()
            detect_human.ALARM = True

# FUNZIONI DI IMAGE ENHANCEMENT
def yuv_equalization(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return image


def adjust_brightness_contrast(image, brightness=0, contrast=0):
    # Normalizzo i valori
    brightness = int((brightness - 0) * (255 - (-255)) / (100 - 0) + (-255))
    contrast = int((contrast - 0) * (127 - (-127)) / (100 - 0) + (-127))

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        image = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)

    if contrast != 0:
        alpha_c = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        gamma_c = 127 * (1 - alpha_c)
        image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)

    return image


def sharpen_image(image):
    sharpened = cv2.filter2D(image,-1,SHARPEN_KERNEL)
    return sharpened


# Applica il Contrast Limited Adaptive Histogram Equalization
def clahe(image):
    # Converto in spazio di colori LAB
    lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab_img)
    # Applico il CLAHE
    clahe = cv2.createCLAHE(2.0,(8,8))
    res_l = clahe.apply(l)
    # Ricostruisco l'immagine e converto in BGR
    res_image = cv2.merge((res_l,a,b))

    return cv2.cvtColor(res_image, cv2.COLOR_LAB2BGR)


def gaussian_blur(image):
    return cv2.GaussianBlur(image, G_KERNEL_SIZE,0)


def enhance_image(image):
    #image = yuv_equalization(image)
    #image = sharpen_image(image)
    #image = gaussian_blur(image)
    image = clahe(image)
    image = sharpen_image(image)
    image = gaussian_blur(image)
    return image


# FUNZIONI PER RIDURRE LA QUALITA' DELL'IMMAGINE
def down_sample(image):
    height, width, c = image.shape
    # Scalo l'immagine ad una risoluzione inferiore
    resized = cv2.resize(image, (int((width/4)), int(height/4)), interpolation=cv2.INTER_LINEAR)
    # Riscalo poi alla risoluzione originale
    reresized = cv2.resize(resized, (int(width), int(height)), interpolation=cv2.INTER_LINEAR)
    return reresized


def add_noise(image):
    # Creo il rumore
    noise = np.random.normal(0,1,image.shape).astype(np.uint8)
    # Sommo il rumore all'immagine
    noise_image = cv2.add(image,noise)

    return noise_image


def reduce_quality(image):
    image = down_sample(image)
    image = add_noise(image)
    return image


def apply_model(frame, human_statistics, animal_statistics, car_statistics):
    # Compute detection on original image
    res = detect(frame)
    res_dict['Human'] = dict_entry(res[res['class'] == 0], (0, 0, 255))
    res_dict['Car'] = dict_entry(res[res['class'] == 2], (0, 255, 0))
    res_dict['Cat'] = dict_entry(res[res['class'] == 15], (255, 0, 0))
    res_dict['Dog'] = dict_entry(res[res['class'] == 16], (255, 0, 0))

    for key, entry in res_dict.items():
        if key in ['Dog', 'Cat']:
            for _, row in entry.value.iterrows():
                if stat:
                    animal_statistics.compute_mean(row['confidence'])
            frame = draw_frame(frame, entry.value, 'Animal', entry.color)
        else:
            match key:
                case 'Human':
                    for _, row in entry.value.iterrows():
                        if stat:
                            human_statistics.compute_mean(row['confidence'])
                case 'Car':
                    for _, row in entry.value.iterrows():
                        if stat:
                            car_statistics.compute_mean(row['confidence'])
            frame = draw_frame(frame, entry.value, key, entry.color)
    return frame

def main():
    global cap
    global stat
    flag = 1
    # Scegliere se analizzare l'input da webcam o un video specifico
    while flag:
        print("############ Video Security System Prototype #############\n\n\n")
        print("\tPlease choose an input\n\t1.Camera source\n\t2.Existing video")
        choice = input()
        match choice:
            case '1':
                cap = cv2.VideoCapture(0)
                flag = 0
            case '2':
                print("\nInsert the video path: ")
                path = input()
                cap = cv2.VideoCapture(path)
                flag = 0
            case _:
                print("\nWrong input!\n")
                continue
    # Scegliere se stampare la confidence misurata
    flag = 1
    while (flag):
        print("\n\tWant to print test statistics?\t\n1.Yes\t\n2.No")
        choice = input()
        match choice:
            case '1':
                stat = 1
                flag = 0
            case '2':
                stat = 0
                flag = 0
            case _:
                print("\nWrong input!\n")
                continue

    # Struttura dati di supporto per la rilevazione degli umani
    sliding_window = deque(maxlen=WINDOW_SIZE)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Applico la detection sull'immagine migliorata
        en_frame = apply_model(enhance_image(frame), human_statistics_enhanced, animal_statistics_enhanced, car_statistics_enhanced)
        cv2.imshow("Enhanced image", en_frame)
        # Applico la detection sull'immagine originale
        or_frame = apply_model(frame, human_statistics_original, animal_statistics_original, car_statistics_original)
        cv2.imshow("Original image", or_frame)
        # Applico la detection sull'immagine a qualità ridotta
        reduced_frame = reduce_quality(frame)
        red_frame = apply_model(reduced_frame, human_statistics_reduced, animal_statistics_reduced, car_statistics_reduced)
        cv2.imshow("Reduced quality image", red_frame)
        en_red_frame = apply_model(enhance_image(reduced_frame), human_statistics_en_red, animal_statistics_en_red, car_statistics_en_red)
        cv2.imshow("Enhanced reduced quality image", en_red_frame)
        # Controlla che un umano sia rilevato per almeno DETECTIONS_TO_ALARM frames
        detect_human(res_dict['Human'].value, sliding_window)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if stat:
        # Stampo le statistiche calcolate
        print("***********ENHANCED STATISTICS**********")
        print(f"-Human confidence: {human_statistics_enhanced.get_mean()}")
        print(f"-Animal confidence: {animal_statistics_enhanced.get_mean()}")
        print(f"-Cars confidence: {car_statistics_enhanced.get_mean()}")


        print("***********ORIGINAL STATISTICS**********")
        print(f"-Human confidence: {human_statistics_original.get_mean()}")
        print(f"-Animal confidence: {animal_statistics_original.get_mean()}")
        print(f"-Cars confidence: {car_statistics_original.get_mean()}")

        print("***********REDUCED QUALITY STATISTICS**********")
        print(f"-Human confidence: {human_statistics_reduced.get_mean()}")
        print(f"-Animal confidence: {animal_statistics_reduced.get_mean()}")
        print(f"-Cars confidence: {car_statistics_reduced.get_mean()}")

        print("***********ENHANCED REDUCED QUALITY STATISTICS**********")
        print(f"-Human confidence: {human_statistics_en_red.get_mean()}")
        print(f"-Animal confidence: {animal_statistics_en_red.get_mean()}")
        print(f"-Cars confidence: {car_statistics_en_red.get_mean()}")
    stop_event.set()
    sound_thread.join()

if __name__ == "__main__":
    main()
