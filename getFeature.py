import cv2
from PIL import Image
from pylab import *

def getFeatureVector(imgPath):
    img1 = cv2.imread(imgPath)
    # 黑白化以便于特征提取
    color1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # 特征提取
    sift=cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
    
    img_1 = cv2.drawKeypoints(color1,keypoints_1,img1)
    cv2.imwrite("res.png",img_1)
    
    return descriptors_1

def getFeatureComparasion(srcPath,tgtPath,compType):
    imgSrc = cv2.imread(srcPath)
    #color1 = cv2.cvtColor(imgSrc, cv2.COLOR_BGR2GRAY)
    imgTgt = cv2.imread(tgtPath)
    #color2 = cv2.cvtColor(imgTgt, cv2.COLOR_BGR2GRAY)
    sift=cv2.SIFT_create(nfeatures=5000)
    keypointsSrc, descriptorsSrc = sift.detectAndCompute(imgSrc,None)
    keypointsTgt, descriptorsTgt = sift.detectAndCompute(imgTgt,None)
    bfMatcher=cv2.BFMatcher()
    if compType==1:
        matches = bfMatcher.knnMatch(descriptorsSrc, descriptorsTgt, k=2)
        # 过滤当前得到的两个最邻近点
        # 当这两个最邻近点之间的临近程度达到一定范围时才认为这两个点相匹配
        good_pts = []
        for res in matches:
            m, n = res
            if m.distance < 0.80 * n.distance:
                good_pts.append([m])
        ret = cv2.drawMatchesKnn(imgSrc, keypointsSrc, imgTgt, keypointsTgt, good_pts, None, flags=2)
    elif compType==2:
        matches = bfMatcher.match(descriptorsSrc, descriptorsTgt)
        matches = sorted(matches, key=lambda x: x.distance)
        ret = cv2.drawMatches(imgSrc, keypointsSrc, imgTgt, keypointsTgt, matches, None, flags=2)
    cv2.imwrite("res.png",ret)


if __name__ == '__main__':
    #getFeatureVector("./static/WechatIMG132.png")
    getFeatureComparasion("./static/WechatIMG132.png","./static/exp.png",1)