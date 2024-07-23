import cv2

# Crear captura de video
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo abrir la cámara. Verifica que esté conectada y que el índice de la cámara sea correcto.")
    exit()

print('Pulsa P para pausar el video y seleccionar el objeto a seguir')

# Función para seleccionar el ROI (región de interés) manualmente
def select_roi(frame):
    r = cv2.selectROI("Selecciona el objeto", frame, fromCenter=False)
    cv2.destroyWindow("Selecciona el objeto")
    return r

# Espera a que se presione la tecla P para seleccionar el objeto
bbox = None
while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo capturar la cámara")
        break

    cv2.imshow("IMAGEN", frame)
    if cv2.waitKey(1) & 0xFF == ord('p'):
        bbox = select_roi(frame)
        break

# Verificar si se seleccionó un ROI válido
if bbox is None or bbox[2] == 0 or bbox[3] == 0:
    print("No se seleccionó un objeto válido.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Inicializar el rastreador de MeanShift
(x, y, w, h) = bbox
track_window = (x, y, w, h)

# Establecer el ROI para el rastreo
roi = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, (0, 60, 32), (180, 255, 255))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Configuración de los criterios de terminación del algoritmo de MeanShift
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # Aplicar el algoritmo MeanShift para encontrar la nueva ubicación
    ret, track_window = cv2.meanShift(dst, track_window, term_crit)

    x, y, w, h = track_window
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
    cv2.imshow("IMAGEN", frame)

    if cv2.waitKey(1) == 27:  # Pulsa 'Esc' para salir
        break

# Liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
