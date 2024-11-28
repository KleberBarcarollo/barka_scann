import cv2
import numpy as np
import utlis

########################################################################
webCamFeed = False  # Altere para True para usar a câmera
pathImage = "C:\\ESTUDOS\\2.jpg"  # Caminho da imagem para processamento
cap = cv2.VideoCapture(0)
cap.set(10, 160)  # Configuração do brilho da câmera
heightImg = 640  # Altura da imagem
widthImg = 480   # Largura da imagem
########################################################################

utlis.initializeTrackbars()  # Inicializa as barras deslizantes para ajustar parâmetros
count = 0

while True:
    if webCamFeed:
        success, img = cap.read()
        if not success:
            print("Erro: Não foi possível capturar a imagem da câmera.")
            break
    else:
        img = cv2.imread(pathImage)  # Lê a imagem do caminho especificado
    img = cv2.resize(img, (widthImg, heightImg))  # Redimensiona a imagem
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # Cria uma imagem em branco para depuração
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Converte a imagem para tons de cinza
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # Aplica desfoque gaussiano
    thres = utlis.valTrackbars()  # Obtém os valores das barras deslizantes para os limiares
    imgThreshold = cv2.Canny(imgBlur, thres[0], thres[1])  # Aplica a detecção de bordas com Canny
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)  # Aplica dilatação
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # Aplica erosão

    # ENCONTRA TODOS OS CONTORNOS
    imgContours = img.copy()  # Copia a imagem para exibição
    imgBigContour = img.copy()  # Copia a imagem para exibição
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Encontra os contornos
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)  # Desenha todos os contornos detectados

    # ENCONTRA O MAIOR CONTORNO
    biggest, maxArea = utlis.biggestContour(contours)  # Encontra o maior contorno
    if biggest.size != 0:
        biggest = utlis.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)  # Desenha o maior contorno
        imgBigContour = utlis.drawRectangle(imgBigContour, biggest, 2)
        pts1 = np.float32(biggest)  # Prepara os pontos para transformação de perspectiva
        pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # Define os pontos de destino
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        # REMOVE 20 PIXELS DE CADA LADO
        imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
        imgWarpColored = cv2.resize(imgWarpColored, (widthImg, heightImg))

        # APLICA LIMIAR ADAPTATIVO
        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre, 3)

        # Array de imagens para exibição
        imageArray = ([img, imgGray, imgThreshold, imgContours],
                      [imgBigContour, imgWarpColored, imgWarpGray, imgAdaptiveThre])

    else:
        imageArray = ([img, imgGray, imgThreshold, imgContours],
                      [imgBlank, imgBlank, imgBlank, imgBlank])

    # RÓTULOS PARA EXIBIÇÃO
    labels = [["Original", "Cinza", "Limiar", "Contornos"],
              ["Maior Contorno", "Perspectiva Corrigida", "Cinza Corrigido", "Limiar Adaptativo"]]

    stackedImage = utlis.stackImages(imageArray, 0.75, labels)
    cv2.imshow("Resultado", stackedImage)

    # SALVA A IMAGEM QUANDO A TECLA 's' FOR PRESSIONADA
    if cv2.waitKey(1) & 0xFF == ord('s'):
        save_path = f"C:\\PYTHON\\SCAN\\SCAN\\myImage_{count}.jpg"
        cv2.imwrite(save_path, imgWarpColored)
        cv2.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50),
                      (1100, 350), (0, 255, 0), cv2.FILLED)
        cv2.putText(stackedImage, "Digitalização Salva", (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.imshow('Resultado', stackedImage)
        cv2.waitKey(300)
        count += 1
