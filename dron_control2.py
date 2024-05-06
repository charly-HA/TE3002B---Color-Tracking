import cv2
import time
import numpy as np
from djitellopy import Tello

# Frame source: 0-> webcam 1-> dron
frame_source = 0

# Iniciar cámara o dron
if frame_source == 0:
    capture = cv2.VideoCapture(0)
elif frame_source == 1:
    drone = Tello()
    drone.connect()
    drone.streamoff()
    drone.streamon()
    drone.left_right_velocity = 0  
    drone.forward_backward_velocity = 0
    drone.up_down_velocity = 0
    drone.yaw_velocity = 0
    drone.status = 0                        # Estado inicial del dron (0 = en tierra)
    drone.height = 0                        # Altura inicial
    drone.speed = 50                        # Velocidad de movimiento inicial
    drone.height_lim = 50

# Tamaño de imagen y área mínima para detectar
h, w = 500, 500
area_min = 800
deadzone = 50

# Valores de HSV para el filtro de color
H_min, H_max = 20, 40
S_min, S_max = 100, 255
V_min, V_max = 100, 255

# Constantes del controlador
kp = 0.9
ki = 0.0
kd = 0.0
kp_a = 0.9
ki_a = 0.0
kd_a = 0.0
ref_area = 29850
area_tolerance = 500

# Callback para los controles
def callback(x):
    pass

# Ventana para los controles
cv2.namedWindow("Control")
cv2.createTrackbar("H_min", "Control", H_min, 179, callback)
cv2.createTrackbar("H_max", "Control", H_max, 179, callback)
cv2.createTrackbar("S_min", "Control", S_min, 255, callback)
cv2.createTrackbar("S_max", "Control", S_max, 255, callback)
cv2.createTrackbar("V_min", "Control", V_min, 255, callback)
cv2.createTrackbar("V_max", "Control", V_max, 255, callback)

def main():
    print("Main program running now")

    # Control manual
    manual = True

    # Variables del controlador
    do_yaw_control = False
    do_vel_x_control = False
    yaw_control = [0, 0]
    error = [0, 0, 0]
    vel_x_control = [0, 0]
    error_area = [0, 0, 0]
    t_now = 0
    t_detect = time.time()*1000//1

    # Ciclo
    while True:
        
        # Actualizar imagen
        if frame_source == 0:
            ret, img = capture.read()
            if not ret:
                print("No se pudo obtener el frame de la webcam.")
                continue
        elif frame_source == 1:
            frame_read = drone.get_frame_read()
            img = frame_read.frame
            if img is None:
                print("No se pudo obtener el frame del dron.")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (500, 500))
        img_tracking = img.copy()

        # HSV de trackbars
        hsv_min = np.array([cv2.getTrackbarPos("H_min", "Control"), cv2.getTrackbarPos("S_min", "Control"), cv2.getTrackbarPos("V_min", "Control")])
        hsv_max = np.array([cv2.getTrackbarPos("H_max", "Control"), cv2.getTrackbarPos("S_max", "Control"), cv2.getTrackbarPos("V_max", "Control")])

        # Filtro de color HSV
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_img, hsv_min, hsv_max)
        result = cv2.bitwise_and(img, img, mask=mask)
        imgBlur = cv2.GaussianBlur(result, (7, 7), 1)
        imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(imgGray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        points_x = []
        areas = []
        for cnt in contours:   
            area = cv2.contourArea(cnt)
            if area > area_min:
                cv2.drawContours(img_tracking, cnt, -1, (255, 0, 255), 7)
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(img_tracking, (cx, cy), 5, (0, 255, 0), -1)
                    cv2.putText(img_tracking, f"({cx}, {cy})", (cx+10, cy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.line(img_tracking, ((w // 2) - deadzone, 0), ((w // 2) - deadzone, w), (255, 255, 0), 2)
                    cv2.line(img_tracking, ((w // 2) + deadzone, 0), ((w // 2) + deadzone, w), (255, 255, 0), 2)
                    points_x.append(cx)
                    areas.append(area)

        # Teclado
        key = cv2.waitKey(1) & 0xFF

        # Uso de dron
        if frame_source == 1:
            
            # Batería, advertencias de nivel bajo y crítico
            cv2.putText(img_tracking, 'Battery:  ' + str(drone.get_battery()), (0, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 3)
            if drone.get_battery() < 25:
                cv2.putText(img_tracking, 'Low level', (0, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
            if drone.get_battery() <= 15:
                cv2.putText(img_tracking, 'Critical level', (0, 90), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

            # Teclas para el modo de control
            if key == 109:          # Presionar m para modo manual
                manual = True
            elif key == 110:        # Presionar n para modo automático
                manual = False
            
            # Control manual
            if manual == True:
                if key == 116:      # Presionar r para despegar
                    if drone.get_battery() >= 25:
                        if drone.status == 0:
                            drone.status = 1
                            drone.takeoff()
                    else:
                        img2 = cv2.imread('fondo.jpg')
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(img2, "Battery low", (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.imshow("Warning", img2)
                elif key == 108:    # Presionar l para aterrizar
                    if drone.status == 1:
                        drone.status = 0
                        drone.land()
                elif key == 104:    # Presionar h para mantener su posición.
                    drone.left_right_velocity = 0 
                    drone.forward_backward_velocity = 0
                    drone.up_down_velocity = 0
                    drone.yaw_velocity = 0
                elif key == 119:    # Presionar w para moverse hacia delante
                    drone.left_right_velocity = 0 
                    drone.forward_backward_velocity = drone.speed
                    drone.up_down_velocity = 0
                    drone.yaw_velocity = 0 
                elif key == 115:    # Presionar s para moverse hacia atrás. 
                    drone.left_right_velocity = 0 
                    drone.forward_backward_velocity = -drone.speed
                    drone.up_down_velocity = 0
                    drone.yaw_velocity = 0
                elif key == 97:     # Presionar a para moverse a la izquierda
                    drone.left_right_velocity = -drone.speed
                    drone.forward_backward_velocity = 0
                    drone.up_down_velocity = 0
                    drone.yaw_velocity = 0
                elif key == 100:    # Presionar d para moverse a la derecha
                    drone.left_right_velocity = drone.speed
                    drone.forward_backward_velocity = 0
                    drone.up_down_velocity = 0
                    drone.yaw_velocity = 0
                elif key == 101:    # Presionar e para subir.
                    drone.left_right_velocity = 0
                    drone.forward_backward_velocity = 0
                    drone.up_down_velocity = drone.speed
                    drone.yaw_velocity = 0
                elif key == 114:    # Presionar r para bajar.
                    drone.left_right_velocity = 0
                    drone.forward_backward_velocity = 0
                    drone.up_down_velocity = -drone.speed
                    drone.yaw_velocity = 0
                elif key == 122:    # Presionar z para girar a la izquierda
                    drone.left_right_velocity = 0
                    drone.forward_backward_velocity = 0
                    drone.up_down_velocity = 0
                    drone.yaw_velocity = -drone.speed
                elif key == 120:    # Presionar x para girar a la derecha
                    drone.left_right_velocity = 0
                    drone.forward_backward_velocity = 0
                    drone.up_down_velocity = 0
                    drone.yaw_velocity = drone.speed
                else:
                    drone.left_right_velocity = 0
                    drone.forward_backward_velocity = 0
                    drone.up_down_velocity = 0
                    drone.yaw_velocity = 0

            # Control automático
            else:
                
                # Obtención del tiempo y diferencia de tiempo
                t_now = time.time()*1000//1
                dt = (t_now - t_detect) / 1000

                # Rotación y desplazamiento al detectar el objeto
                if len(points_x) != 0:
                    t_detect = t_now

                    # Obtención del centro del objeto
                    x = np.median(points_x)

                    # Obtención del área del objeto
                    area_obj = max(areas)
                    print("area: ", area_obj)

                    # Realizar rotación con controlador
                    if x < ((w // 2) - deadzone) or x > ((w // 2) + deadzone):
                        do_yaw_control = True
                        do_vel_x_control = False

                        # Cálculo de velocidad de yaw con controlador
                        error[0] = (w // 2) - x
                        yaw_control[0] = int((kp + ki * dt + kd / dt) * error[0] - (kp + 2.0 * kd / dt) * error[1] + kd / dt * error[2] + yaw_control[1])
                        error[2] = error[1]
                        error[1] = error[0]
                        yaw_control[1] = yaw_control[0]
                    
                    # Realizar desplazamiento con controlador
                    elif area_obj < (ref_area - area_tolerance) or area_obj > (ref_area + area_tolerance):
                        do_yaw_control = False
                        do_vel_x_control = True

                        # Cálculo de velocidad de x con controlador
                        error_area[0] = ref_area - area_obj
                        vel_x_control[0] = int((kp_a + ki_a * dt + kd_a / dt) * error_area[0] - (kp_a + 2.0 * kd_a / dt) * error_area[1] + kd_a / dt * error_area[2] + vel_x_control[1])
                        error_area[2] = error_area[1]
                        error_area[1] = error_area[0]
                        vel_x_control[1] = vel_x_control[0]
                    
                    else:

                        # Detener rotación
                        do_yaw_control = False

                        # Detener desplazamiento
                        do_vel_x_control = False

                # Rotación y desplazamiento al no detectar el objeto
                else:

                    # Detener rotación y desplazamiento después de 3 segundos, si hay
                    if dt >= 3:
                        if do_yaw_control == True:
                            do_yaw_control = False
                        if do_vel_x_control == True:
                            do_vel_x_control = False

                # Establecer valores de velocidad
                if do_yaw_control:
                    drone.yaw_velocity = yaw_control[0]
                    print(yaw_control[0])
                else:
                    drone.yaw_velocity = 0
                if do_vel_x_control:
                    drone.forward_backward_velocity = vel_x_control[0]
                    print(vel_x_control[0])
                else:
                    drone.forward_backward_velocity = 0
                drone.left_right_velocity = 0
                drone.up_down_velocity = 0
            
            # Enviar control
            drone.send_rc_control(drone.left_right_velocity, drone.forward_backward_velocity, drone.up_down_velocity, drone.yaw_velocity)           

        # Mostrar imagen
        cv2.imshow("Image", img_tracking)
        
        # Finalizar
        if key == 113:              # Presionar q para finalizar
            cv2.destroyAllWindows()
            if frame_source == 1:
                drone.streamoff()
                drone.end()
            break

try:
    main()
except KeyboardInterrupt:
    print('KeyboardInterrupt exception is caught')
    cv2.destroyAllWindows()
    if frame_source == 1:
        drone.land()
        drone.streamoff()
        drone.end()
else:
    print('No exceptions are caught')

    #pip install opencv-python djitellopy numpy
