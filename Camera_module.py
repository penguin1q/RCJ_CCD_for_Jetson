# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time

xmin, xmax = 280, 1000
ymin, ymax = 0, 720
px_size = 720

# JetsonではCSIカメラの画像をGSteamerでキャプチャする以下のように条件を指定する
# appsinkは必須ではないがロボカップで使う場合指定したほうがいい
# max-buffersは保存するバッファ数(デフォ3)
# dropはフレームレート以下の処理速度の時フローしたデータを捨てるか否か(デフォfalse)
GST_STR = 'nvarguscamerasrc saturation=1.1\
		! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, framerate=(fraction)60/1 \
		! nvvidconv ! video/x-raw, width=(int)1280, height=(int)720, format=(string)BGRx \
		!videobalance  \
        ! videoconvert \
		! appsink max-buffers=1 drop=True'

# 画像の2極化(欲しい色だけモノクロに)
# 輪郭を抽出する
# 凸包処理(かけた部分を修正)
# 塊ごとに四角で囲む
class Find_rect:
    def __init__(self, hsv_min, hsv_max):
        self.hsv_min = hsv_min
        self.hsv_max = hsv_max

    def find_rect_find_target_color(self, hsv):
        # 2極化
        if self.hsv_max < self.hsv_min:
            mask_image1 = cv2.inRange(hsv, (0, self.hsv_min[1], self.hsv_min[2]), tuple(self.hsv_max))
            mask_image2 = cv2.inRange(hsv, tuple(self.hsv_min), (255, self.hsv_max[1], self.hsv_max[2]))
            mask_image = mask_image1 + mask_image2
        else :
            mask_image = cv2.inRange(hsv, tuple(self.hsv_min), tuple(self.hsv_max))
        
        # 輪郭を抽出
        contours,_ =cv2.findContours(mask_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rects = []
        # 凸包処理をして，四角で囲む
        for contour in contours:
            approx = cv2.convexHull(contour)
            rect = cv2.boundingRect(approx)
            rects.append(np.array(rect))
        return rects

# 画像を取得し，色を検出して表示，座標を返す
class Camera_Module: 
    def __init__(self):
        self.orange = Find_rect([247, 97, 185], [12, 228, 255])
        self.capture = cv2.VideoCapture(GST_STR, cv2.CAP_GSTREAMER)
        self.orange_rects = []
        self.orange_rect = []
        self.pretime = 0
        self.orange_min_size = 10
    

    def read_pos_data(self, dposx, dposy):
        # 画像を取得
        _, frame = self.capture.read()
        # BGRからHSVに変換
        hsv = cv2.cvtColor(frame[:,xmin:xmax], cv2.COLOR_BGR2HSV_FULL)
        
        # Find_rectのclassを使い，色の四角群を取得
        orange_rects = self.orange.find_rect_find_target_color(hsv)
        # 四角群の中で一番大きいものを選択する(改善の余地あり)
        if orange_rects:
            self.orange_rect = max(orange_rects, key=(lambda x: x[2] * x[3]))
        else:
            self.orange_rect = []
        
        # 画像に四角の線を書く       
        if len(self.orange_rect) >0:
            cv2.rectangle(frame[:,xmin:xmax], tuple(self.orange_rect[0:2]), tuple(self.orange_rect[0:2] + self.orange_rect[2:4]), (0, 0, 255), thickness=2)
        cv2.imshow('result', frame[:,xmin:xmax])
        
        orange_rect_data=[]

        # 全方位ミラーように原点中心のデータに変換する        
        if len(self.orange_rect) > 0:
            orange_rect_data = [int(self.orange_rect[0] + self.orange_rect[2] / 2 - (px_size/2) + dposx),
                                int(self.orange_rect[1] + self.orange_rect[3] / 2 - (px_size/2) + dposy),
                                int(self.orange_rect[2]), int(self.orange_rect[3])]
            
            # 最小サイズより小さい時ノイズとして無視
            if (self.orange_rect[2] * self.orange_rect[3]) < self.orange_min_size:
                orange_rect_data = []
        else:
            orange_rect_data = []

        dt = time.perf_counter() - self.pretime
        self.pretime = time.perf_counter()

        return orange_rect_data
        
    def __del__(self):
        self.capture.release()
        cv2.destroyAllWindows()


def main():
    camera = Camera_Module()
    pretime = 0
    try:
        while cv2.waitKey(1) < 0:
            dt = time.perf_counter() - pretime
            pretime = time.perf_counter()
            orange_rect= camera.read_pos_data()
            
            print(f"{dt:.3} {1/dt:.3} {orange_rect}")
    
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
	main() 