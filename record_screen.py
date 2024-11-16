import win32gui, win32ui, win32con
from PIL import Image
import numpy as np
import cv2

def capture_window(window_title:str) -> np.ndarray:
   hwnd = win32gui.FindWindow(None, window_title)
   if not hwnd:
      raise Exception(f"Window '{window_title}' not found!")

   left, top, right, bottom = win32gui.GetWindowRect(hwnd)
   width = right - left
   height = bottom - top

   hwnd_dc = win32gui.GetWindowDC(hwnd)
   mfc_dc  = win32ui.CreateDCFromHandle(hwnd_dc)
   save_dc = mfc_dc.CreateCompatibleDC()

   bitmap = win32ui.CreateBitmap()
   bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
   save_dc.SelectObject(bitmap)
   save_dc.BitBlt((0,0), (width,height), mfc_dc, (0,0), win32con.SRCCOPY)

   bmp_info = bitmap.GetInfo()
   bmp_str  = bitmap.GetBitmapBits(True)
   img = Image.frombuffer('RGB', (bmp_info["bmWidth"],bmp_info["bmHeight"]), bmp_str, 'raw', 'BGRX', 0, 1)

   win32gui.DeleteObject(bitmap.GetHandle())
   save_dc.DeleteDC()
   mfc_dc.DeleteDC()
   win32gui.ReleaseDC(hwnd, hwnd_dc)

   return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def list_window_titles():
   def callback(hwnd, windows):
      if win32gui.IsWindowVisible(hwnd):
         window_text = win32gui.GetWindowText(hwnd)
         if window_text:  # Only add windows with titles
               windows.append((hwnd, window_text))
      return True
    
   windows = []
   win32gui.EnumWindows(callback, windows)
   
   # Sort windows by title for easier reading
   windows.sort(key=lambda x: x[1].lower())
   
   print("\nActive Windows:")
   print("-" * 50)
   for hwnd, title in windows:
      print(f"Handle: {hwnd:10} | Title: {title}")

if __name__ == "__main__":
   import argparse
   parser = argparse.ArgumentParser()
   parser.add_argument('window_title', type=str)
   args = parser.parse_args()

   if args.window_title == "list":
      list_window_titles()
   else:
      im = capture_window(args.window_title)
      cv2.imshow("frame", im)
      cv2.waitKey()
