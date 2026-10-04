import cv2
from managers import WindowManager, CaptureManager
import filters
import depth
import rects
from trackers import FaceTracker


class Cameo(object):
    def __init__(self):
        self._windowManager = WindowManager('Cameo',
                                            self.onKeyPress)
        self._captureManager = CaptureManager(cv2.VideoCapture(0),
                                              self._windowManager, True)
        self._curveFilter = filters.BGRPortraCurveFilter()

        self._faceTracker = FaceTracker()
        self._shouldDrawDebugRects = False

    def run(self):
        """Run the main loop"""
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame

            if frame is not None:
                self._faceTracker.update(frame)
                faces = self._faceTracker.faces
                rects.swapRects(frame, frame,
                                [face.faceRect for face in faces])


                filters.strokeEdges(frame, frame)
                self._curveFilter.apply(frame, frame)
                if self._shouldDrawDebugRects:
                    self._faceTracker.drawDebugRects(frame)
            
            self._captureManager.exitFrame()
            self._windowManager.processEvents()


    def onKeyPress(self, keycode):
        """Handle a keypress.
        space -> Take a screenshot.
        tab ->  Start/stop recording a screenshot.
        escape ->  Quit.
        """

        if keycode == 32:
            self._captureManager.writeImage('screenshot.png')
        elif keycode == 9:
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo('screencast.avi')
            else:
                self._captureManager.stopWritingVideo()
        elif keycode == 120: # x
                self._shouldDrawDebugRects = \
                    not self._shouldDrawDebugRects
        elif keycode == 27:
            self._windowManager.destroyWindow()


class CameoDepth(Cameo):
    def __init__(self):
        self._windowManager = WindowManager('Cameo', self.onKeyPress)
        #device = cv2.CAP_OPENNI2
        device = cv2.CAP_OPENNI2_ASUS

        self._captureManager = CaptureManager(
            cv2.VideoCapture(device), self._windowManager, True
        )

        self._faceTracker = FaceTracker()
        self._shouldDrawDebugRects = False
        self._curveFilter = filters.BGRPortraCurveFilter()

    def run(self):
        """Run the main loop"""
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            self._captureManager.channel = cv2.CAP_OPENNI_DISPARITY_MAP
            disparityMap = self._captureManager.frame
            self._captureManager.channel = cv2.CAP_OPENNI_VALID_DEPTH_MASK
            validDepthMask = self._captureManager.frame
            self._captureManager.channel = cv2.CAP_OPENNI_BGR_IMAGE

            frame = self._captureManager.frame
            if frame is None:
                #Failed to capture a BGR frame
                #Try to capture an infrared frame instead
                self._captureManager.channel = cv2.CAP_OPENNI_IR_IMAGE
                frame = self._captureManager.frame

            if frame is not None:
                self._faceTracker.update(frame)
                faces = self._faceTracker.faces
                masks = [
                    depth.createMedianMask(
                        disparityMap, validDepthMask, face.faceRect) \
                    for face in faces
                ]
                rects.swapRects(frame, frame,
                                [face.faceRect for face in faces], masks)

                if self._captureManager.channel == cv2.CAP_OPENNI_BGR_IMAGE:
                    # A BGR frame was captured.
                    # Apply filters to it.
                    filters.strokeEdges(frame, frame)
                    self._curveFilter.apply(frame, frame)

                if self._shouldDrawDebugRects:
                    self._faceTracker.drawDebugRects(frame)

            self._captureManager.exitFrame()
            self._windowManager.processEvents()




if __name__ == "__main__":
    #Cameo().run()
    CameoDepth().run()