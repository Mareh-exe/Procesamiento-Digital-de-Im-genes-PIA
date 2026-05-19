import cv2
import numpy as np
import pydicom

from tkinter import *
from tkinter import filedialog

from PIL import Image, ImageTk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# =========================
# VARIABLES GLOBALES
# =========================

img_original = None
img_gray = None

# =========================
# MOSTRAR IMAGEN
# =========================

def mostrar_imagen(imagen, panel):

    imagen = cv2.resize(imagen, (350, 350))

    if len(imagen.shape) == 2:
        imagen = Image.fromarray(imagen)
    else:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        imagen = Image.fromarray(imagen)

    imagen_tk = ImageTk.PhotoImage(imagen)

    panel.config(image=imagen_tk)
    panel.image = imagen_tk


# =========================
# CARGAR IMAGEN
# =========================

def cargar_imagen():

    global img_original, img_gray

    ruta = filedialog.askopenfilename(
        filetypes=[
            ("Archivos DICOM", "*.dcm"),
            ("Imagenes", "*.png *.jpg *.jpeg")
        ]
    )

    if ruta:

        # DICOM
        if ruta.endswith(".dcm"):

            dicom = pydicom.dcmread(ruta)

            img_gray = dicom.pixel_array

            img_gray = cv2.normalize(
                img_gray,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            )

            img_gray = np.uint8(img_gray)

        # PNG/JPG
        else:

            imagen = cv2.imread(ruta)

            img_gray = cv2.cvtColor(
                imagen,
                cv2.COLOR_BGR2GRAY
            )

        # Convertir para dibujar
        img_original = cv2.cvtColor(
            img_gray,
            cv2.COLOR_GRAY2BGR
        )

        mostrar_imagen(img_gray, panel_original)


# =========================
# HISTOGRAMA
# =========================

def ecualizar_histograma():

    global img_gray

    if img_gray is None:
        return

    ecualizada = cv2.equalizeHist(img_gray)

    mostrar_imagen(ecualizada, panel_resultado)

    fig, axs = plt.subplots(1,2, figsize=(8,4))

    axs[0].hist(img_gray.ravel(), 256, [0,256])
    axs[0].set_title("Histograma Original")

    axs[1].hist(ecualizada.ravel(), 256, [0,256])
    axs[1].set_title("Histograma Ecualizado")

    ventana = Toplevel(root)

    canvas = FigureCanvasTkAgg(fig, master=ventana)

    canvas.draw()

    canvas.get_tk_widget().pack()


# =========================
# FILTRO MEDIA
# =========================

def filtro_media():

    global img_gray

    if img_gray is None:
        return

    media = cv2.blur(img_gray, (7,7))

    mostrar_imagen(media, panel_resultado)


# =========================
# FILTRO GAUSSIANO
# =========================

def filtro_gaussiano():

    global img_gray

    if img_gray is None:
        return

    gauss = cv2.GaussianBlur(img_gray, (9,9), 0)

    mostrar_imagen(gauss, panel_resultado)


# =========================
# FILTRO SOBEL
# =========================

def filtro_sobel():

    global img_gray

    if img_gray is None:
        return

    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)

    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

    sobel = cv2.magnitude(sobelx, sobely)

    sobel = np.uint8(sobel)

    mostrar_imagen(sobel, panel_resultado)


# =========================
# FILTRO LAPLACIANO
# =========================

def filtro_laplaciano():

    global img_gray

    if img_gray is None:
        return

    lap = cv2.Laplacian(img_gray, cv2.CV_64F)

    lap = np.uint8(np.absolute(lap))

    mostrar_imagen(lap, panel_resultado)


# =========================
# RESTA DE IMAGENES
# =========================

def resta_imagenes():

    global img_gray

    if img_gray is None:
        return

    suavizada = cv2.GaussianBlur(img_gray, (11,11), 0)

    resta = cv2.subtract(img_gray, suavizada)

    mostrar_imagen(resta, panel_resultado)


# =========================
# OPERACIONES LOGICAS
# =========================

def operaciones_logicas():

    global img_gray

    if img_gray is None:
        return

    _, mascara = cv2.threshold(
        img_gray,
        120,
        255,
        cv2.THRESH_BINARY
    )

    resultado = cv2.bitwise_and(
        img_gray,
        mascara
    )

    mostrar_imagen(resultado, panel_resultado)


# =========================
# SEGMENTACION
# =========================

def actualizar_umbral(valor):

    global img_gray

    if img_gray is None:
        return

    umbral = int(valor)

    _, binaria = cv2.threshold(
        img_gray,
        umbral,
        255,
        cv2.THRESH_BINARY
    )

    mostrar_imagen(binaria, panel_resultado)


# =========================
# FOURIER
# =========================

def transformada_fourier():

    global img_gray

    if img_gray is None:
        return

    f = np.fft.fft2(img_gray)

    fshift = np.fft.fftshift(f)

    magnitud = 20 * np.log(np.abs(fshift) + 1)

    rows, cols = img_gray.shape

    crow, ccol = rows // 2, cols // 2

    mask = np.zeros((rows, cols), np.uint8)

    r = 40

    cv2.circle(mask, (ccol, crow), r, 1, -1)

    fshift_filtrado = fshift * mask

    f_ishift = np.fft.ifftshift(fshift_filtrado)

    img_back = np.fft.ifft2(f_ishift)

    img_back = np.abs(img_back)

    fig, axs = plt.subplots(1,3, figsize=(12,4))

    axs[0].imshow(img_gray, cmap='gray')
    axs[0].set_title("Original")

    axs[1].imshow(magnitud, cmap='gray')
    axs[1].set_title("Espectro Fourier")

    axs[2].imshow(img_back, cmap='gray')
    axs[2].set_title("Filtrado")

    for ax in axs:
        ax.axis("off")

    ventana = Toplevel(root)

    canvas = FigureCanvasTkAgg(fig, master=ventana)

    canvas.draw()

    canvas.get_tk_widget().pack()


# =========================
# DETECCION DE TUMOR
# =========================

def detectar_tumor():

    global img_original, img_gray

    if img_gray is None:
        return

    resultado = img_original.copy()

    # Suavizar
    blur = cv2.GaussianBlur(img_gray, (5,5), 0)

    # Ecualizar
    ecualizada = cv2.equalizeHist(blur)

    # Detectar zonas brillantes
    _, thresh = cv2.threshold(
        ecualizada,
        185,
        255,
        cv2.THRESH_BINARY
    )

    # Operaciones morfologicas
    kernel = np.ones((5,5), np.uint8)

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel
    )

    thresh = cv2.dilate(
        thresh,
        kernel,
        iterations=2
    )

    # Encontrar contornos
    contornos, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    altura, ancho = img_gray.shape

    for c in contornos:

        area = cv2.contourArea(c)

        # Ignorar ruido pequeño
        if area < 300:
            continue

        x, y, w, h = cv2.boundingRect(c)

        # Ignorar bordes
        if x < 40 or y < 40:
            continue

        if x + w > ancho - 40:
            continue

        if y + h > altura - 40:
            continue

        # Relacion ancho/alto
        relacion = w / h

        # Filtrar formas raras
        if relacion < 0.5 or relacion > 2:
            continue

        # Dibujar contorno
        cv2.drawContours(
            resultado,
            [c],
            -1,
            (255,255,0),
            2
        )

        # Rectangulo
        cv2.rectangle(
            resultado,
            (x,y),
            (x+w, y+h),
            (0,0,255),
            2
        )

        # Texto
        cv2.putText(
            resultado,
            "Posible Tumor",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,0,255),
            2
        )

    mostrar_imagen(resultado, panel_resultado)


# =========================
# INTERFAZ
# =========================

root = Tk()

root.title("Procesamiento de Imagenes Medicas")

root.geometry("1100x800")

root.config(bg="#DDEEFF")

titulo = Label(
    root,
    text="Procesamiento de Imagenes Medicas",
    font=("Arial", 22, "bold"),
    bg="#DDEEFF"
)

titulo.pack(pady=10)

frame_imagenes = Frame(root, bg="#DDEEFF")

frame_imagenes.pack()

panel_original = Label(frame_imagenes, bg="white")

panel_original.grid(row=0, column=0, padx=20)

panel_resultado = Label(frame_imagenes, bg="white")

panel_resultado.grid(row=0, column=1, padx=20)

frame_botones = Frame(root, bg="#DDEEFF")

frame_botones.pack(pady=20)

Button(
    frame_botones,
    text="Cargar Imagen",
    command=cargar_imagen,
    width=22,
    bg="#4CAF50",
    fg="white"
).grid(row=0, column=0, padx=5, pady=5)

Button(
    frame_botones,
    text="Ecualizar Histograma",
    command=ecualizar_histograma,
    width=22
).grid(row=0, column=1, padx=5, pady=5)

Button(
    frame_botones,
    text="Filtro Media",
    command=filtro_media,
    width=22
).grid(row=0, column=2, padx=5, pady=5)

Button(
    frame_botones,
    text="Filtro Gaussiano",
    command=filtro_gaussiano,
    width=22
).grid(row=1, column=0, padx=5, pady=5)

Button(
    frame_botones,
    text="Filtro Sobel",
    command=filtro_sobel,
    width=22
).grid(row=1, column=1, padx=5, pady=5)

Button(
    frame_botones,
    text="Filtro Laplaciano",
    command=filtro_laplaciano,
    width=22
).grid(row=1, column=2, padx=5, pady=5)

Button(
    frame_botones,
    text="Resta de Imagenes",
    command=resta_imagenes,
    width=22
).grid(row=2, column=0, padx=5, pady=5)

Button(
    frame_botones,
    text="Operaciones Logicas",
    command=operaciones_logicas,
    width=22
).grid(row=2, column=1, padx=5, pady=5)

Button(
    frame_botones,
    text="Transformada Fourier",
    command=transformada_fourier,
    width=22
).grid(row=2, column=2, padx=5, pady=5)

Button(
    frame_botones,
    text="Detectar Tumor",
    command=detectar_tumor,
    width=22,
    bg="#FF5252",
    fg="white"
).grid(row=3, column=1, padx=5, pady=5)

texto_slider = Label(
    root,
    text="Ajuste de Umbral",
    font=("Arial", 16),
    bg="#DDEEFF"
)

texto_slider.pack(pady=10)

slider = Scale(
    root,
    from_=0,
    to=255,
    orient=HORIZONTAL,
    length=500,
    command=actualizar_umbral
)

slider.pack(pady=10)

root.mainloop()
