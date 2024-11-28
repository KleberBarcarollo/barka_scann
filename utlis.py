import cv2
import numpy as np

## EMPILHAR TODAS AS IMAGENS EM UMA JANELA
def stackImages(imgArray, scale, lables=[]):
    # Obter número de linhas e colunas no array de imagens
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)  # Verifica se as linhas contêm sublistas
    width = imgArray[0][0].shape[1]  # Largura das imagens
    height = imgArray[0][0].shape[0]  # Altura das imagens

    if rowsAvailable:
        # Redimensionar e ajustar imagens se houver várias sublistas
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:  # Converter imagens em escala de cinza para BGR
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)

        # Criar imagens em branco para alinhamento
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows

        # Concatenar horizontalmente as imagens
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])

        # Concatenar verticalmente as imagens
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        # Redimensionar imagens se não houver sublistas
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2:  # Converter escala de cinza para BGR
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)

        # Empilhar horizontalmente
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor

    if len(lables) != 0:  # Adicionar rótulos às imagens se fornecidos
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        for d in range(0, rows):
            for c in range(0, cols):
                cv2.rectangle(ver, (c * eachImgWidth, eachImgHeight * d),
                              (c * eachImgWidth + len(lables[d][c]) * 13 + 27, 30 + eachImgHeight * d),
                              (255, 255, 255), cv2.FILLED)  # Criar retângulos para os rótulos
                cv2.putText(ver, lables[d][c], (eachImgWidth * c + 10, eachImgHeight * d + 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)  # Adicionar texto aos rótulos
    return ver


# REORDENAR PONTOS DE UM CONTORNO
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))  # Reestruturar os pontos para 4x2
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)  # Criar uma matriz de zeros
    add = myPoints.sum(1)  # Somar as coordenadas (x + y)
    myPointsNew[0] = myPoints[np.argmin(add)]  # Obter ponto superior esquerdo
    myPointsNew[3] = myPoints[np.argmax(add)]  # Obter ponto inferior direito
    diff = np.diff(myPoints, axis=1)  # Calcular diferenças (x - y)
    myPointsNew[1] = myPoints[np.argmin(diff)]  # Obter ponto superior direito
    myPointsNew[2] = myPoints[np.argmax(diff)]  # Obter ponto inferior esquerdo

    return myPointsNew


# OBTER O MAIOR CONTORNO
def biggestContour(contours):
    biggest = np.array([])  # Inicializar maior contorno
    max_area = 0  # Inicializar área máxima

    for i in contours:
        area = cv2.contourArea(i)  # Calcular área do contorno
        if area > 5000:  # Verificar se a área é significativa
            peri = cv2.arcLength(i, True)  # Calcular o perímetro do contorno
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)  # Aproximar o contorno para simplificação
            if area > max_area and len(approx) == 4:  # Verificar se é um quadrilátero maior
                biggest = approx  # Atualizar maior contorno
                max_area = area  # Atualizar área máxima

    return biggest, max_area


# DESENHAR UM RETÂNGULO USANDO PONTOS
def drawRectangle(img, biggest, thickness):
    # Desenhar linhas conectando os pontos do retângulo
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)

    return img


# FUNÇÃO NULA PARA TRACKBARS
def nothing(x):
    pass


# INICIALIZAR TRACKBARS
def initializeTrackbars(initialTracbarVals=0):
    cv2.namedWindow("Trackbars")  # Criar janela de controles
    cv2.resizeWindow("Trackbars", 360, 240)  # Redimensionar a janela
    cv2.createTrackbar("Threshold1", "Trackbars", 200, 255, nothing)  # Criar trackbar para Threshold1
    cv2.createTrackbar("Threshold2", "Trackbars", 200, 255, nothing)  # Criar trackbar para Threshold2


# OBTER VALORES DOS TRACKBARS
def valTrackbars():
    Threshold1 = cv2.getTrackbarPos("Threshold1", "Trackbars")  # Obter valor do Threshold1
    Threshold2 = cv2.getTrackbarPos("Threshold2", "Trackbars")  # Obter valor do Threshold2
    src = Threshold1, Threshold2  # Retornar como tupla
    return src
