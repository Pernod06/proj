import cv2
from Cython.Compiler.Naming import frame_code_cname


def take_video_frame(in_path, out_path_prefix):
    cap = cv2.VideoCapture(in_path)
    c = 1
    frameRate = 10
    while True:
        ret, frame = cap.read()
        if ret:
            if c % frameRate == 0:
                out_path = f"{out_path_prefix}_frame_{c}.jpg"
                cv2.imwrite(out_path, frame)
            c += 1
        else:
            break
    cap.release()

in_path = "../车牌识别视频/2.mp4"
out_path = "out_path/"


if __name__ == "__main__":

    take_video_frame(in_path, out_path)