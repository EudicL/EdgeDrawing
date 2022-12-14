import os
import cv2
import numpy as np
from PIL import Image
import sys   
sys.setrecursionlimit(100000) # default 1000  

class EdgeDrawing:


    def __init__(self) -> None:
        self.Mag = []
        self.Direct = []

        self.anchor = []
        self.edges = []
        self.height = 0
        self.width = 0
        self.sobel_threshold = 20
        self.threshold = 10
        self.ver = 1
        self.hor = 2
     

    def getAnchor(self, image):

        if image is None:
            return

        img = image.copy()
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

       
        # gauss = cv2.getGaussianKernel((5,5),1)
        # img_gau = cv2.GaussianBlur(img, (5,5), 1,1)


        # ret, binary = cv2.threshold(img,50, 255, cv2.THRESH_BINARY_INV) 

        # # kernel = np.ones((3, 3), np.uint8)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # binary = cv2.dilate(binary, kernel, 1)

        # binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        dst = img.copy() #cv2.bitwise_and(img, binary)

        Gx = cv2.Sobel(dst, cv2.CV_32F, 1,0,3)
        Gy = cv2.Sobel(dst, cv2.CV_32F,0,1,3)


        self.height, self.width = dst.shape

        height, width = dst.shape
       

        self.Mag = np.zeros((height, width),dtype=np.float64)
        self.Direct = np.zeros((height, width),dtype=np.float64)
        for i in range(1,height-1):
            for j in range(1,width-1):
                gx = abs(Gx[i][j])
                gy = abs(Gy[i][j])

                if gx + gy > self.sobel_threshold:
                    self.Mag[i][j] = gx + gy
                
                if gx >= gy:
                    self.Direct[i][j] = self.hor
                else:
                    self.Direct[i][j] = self.ver

        # self.Mag = cv2.bitwise_and(self.Mag, self.Mag, mask = binary)
        # self.Direct = cv2.bitwise_and(self.Direct, self.Direct, mask = binary)


        for i in range(1,self.height-1):
            for j in range(1,self.width-1):

                if self.Direct[i][j] == 1:
                    if (self.Mag[i][j] - self.Mag[i][j-1]) >= self.threshold and (self.Mag[i][j] - self.Mag[i][j+1]) >= self.threshold:
                        self.anchor.append((i,j))
                    
                elif self.Direct[i][j] == 2:
                    if (self.Mag[i][j] - self.Mag[i-1][j]) >= self.threshold and (self.Mag[i][j] - self.Mag[i+1][j]) >= self.threshold:
                        self.anchor.append((i,j))

        print(len(self.anchor))

    def getEdges(self):

        isEdge = np.zeros((self.height,self.width))

        for i in range(0, len(self.anchor)):
            edge = []
            x = self.anchor[i][0]
            y = self.anchor[i][1]
            self.searchFromAnchor(x,y,isEdge, edge)
            if len(edge)>0:
                self.edges.append(edge)


    def searchFromAnchor(self, x,y,isEdge, edge):
        if x-1 <0 or y -1 < 0  or x+1>self.height or y+1 > self.width :
            return
        
        if self.Mag[x][y] > 0  and isEdge[x][y] == 0:
            edge.append((x,y))
            isEdge[x][y] = 1
            

            if self.Direct[x][y] == self.hor :
                if isEdge[x-1][y-1] == 0 and isEdge[x-1][y] == 0 and isEdge[x-1][y+1] == 0:
                    if self.Mag[x-1][y-1] > self.Mag[x-1][y] and self.Mag[x-1][y-1] > self.Mag[x-1][y+1]:
                        self.searchFromAnchor(x-1,y-1, isEdge, edge)
                    elif self.Mag[x-1][y+1] > self.Mag[x-1][y] and self.Mag[x-1][y+1] > self.Mag[x-1][y-1]:
                        self.searchFromAnchor(x-1,y+1, isEdge, edge)
                    else:
                        self.searchFromAnchor(x-1,y, isEdge, edge)


                if isEdge[x+1][y-1] == 0 and isEdge[x+1][y] == 0 and isEdge[x+1][y+1] == 0:
                    if self.Mag[x+1][y-1] > self.Mag[x+1][y] and self.Mag[x+1][y-1] > self.Mag[x+1][y+1]:
                        self.searchFromAnchor(x+1,y-1, isEdge, edge)
                    elif self.Mag[x+1][y+1] > self.Mag[x+1][y] and self.Mag[x+1][y+1] > self.Mag[x+1][y-1]:
                        self.searchFromAnchor(x+1,y+1, isEdge, edge)
                    else:
                        self.searchFromAnchor(x+1,y, isEdge, edge)

            if self.Direct[x][y] == self.ver:
                    if isEdge[x+1][y-1] == 0 and isEdge[x][y-1] == 0 and isEdge[x-1][y-1] == 0:
                        if self.Mag[x-1][y-1] > self.Mag[x][y-1] and self.Mag[x-1][y-1] > self.Mag[x+1][y-1]:
                            self.searchFromAnchor(x-1,y-1, isEdge, edge)
                        elif self.Mag[x+1][y-1] > self.Mag[x-1][y-1] and self.Mag[x+1][y-1] > self.Mag[x][y-1]:
                            self.searchFromAnchor(x+1,y-1, isEdge, edge)
                        else:
                            self.searchFromAnchor(x,y-1, isEdge, edge)


                    if isEdge[x+1][y+1] == 0 and isEdge[x][y+1] == 0 and isEdge[x-1][y+1] == 0:
                        if self.Mag[x-1][y+1] > self.Mag[x][y+1] and self.Mag[x-1][y+1] > self.Mag[x+1][y+1]:
                            self.searchFromAnchor(x-1,y+1, isEdge, edge)
                        elif self.Mag[x+1][y+1] > self.Mag[x][y+1] and self.Mag[x+1][y+1] > self.Mag[x-1][y+1]:
                            self.searchFromAnchor(x+1,y+1, isEdge, edge)
                        else:
                            self.searchFromAnchor(x,y+1, isEdge, edge)

    def getEdgeImage(self, img):
        self.getAnchor(img)
        self.getEdges()
        res = img.copy()
        print(len(self.edges))
        return self.edges

        for i in range(0, len(self.edges)):
            for k in range(0, len(self.edges[i])):
                m,n = self.edges[i][k]
                print(m,n)
                res[m][n] = 255

        return res


if __name__ == "__main__":


    image_path = "./demo.png"
    ED = EdgeDrawing()

    img = cv2.imread(image_path)

    img_copy = img.copy()

    res = img.copy()
    height, width = res.shape[:2]

    res = cv2.resize(res, None, fx=0.5, fy=0.5)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Gaussian
    img = cv2.pyrDown(img) 

    edges = ED.getEdgeImage(img)


    
    for i in range(0, len(edges)):
        bgr = np.random.randint(0,255,3,dtype=np.int32)
        for k in range(0, len(edges[i])):
            m,n = edges[i][k]
            
            res[m][n] = (bgr[0],bgr[1],bgr[2])
    
    cv2.imshow("1", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    



    


    # random color

    # bgr = np.random.randint(0,255,3,dtype=np.int32)
    # for k in range(0, len(edges[max_index])):
    #     m,n = edges[max_index][k]
        
    #     res[m][n] = (bgr[0],bgr[1],bgr[2])



    # cv2.rectangle(img_copy, (y0,x0),(y1,x1), (0,0,255), 2)
    # cv2.imshow("1", img_copy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()





    