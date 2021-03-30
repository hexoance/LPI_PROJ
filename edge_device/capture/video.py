# From http://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
import cv2
import time
import datetime
from threading import Thread


class WebcamVideoStream:
    def __init__(self, src, width, height, in_q, out_q):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()

        self.in_q = in_q
        self.out_q = out_q

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()

        # start the thread to read frames from the video stream, and the thread to process the frames
        Thread(target=self.update, args=()).start()

        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:  # fps._numFrames < 120
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (grabbed, frame) = self.stream.read()

            # increment the total number of frames examined during the
            # start and end intervals
            self._numFrames += 1

            item = {'type': 'video', 'data': frame}
            self.in_q.put(item)

            t = time.time()

            output_rgb = cv2.cvtColor(self.out_q.get(), cv2.COLOR_RGB2BGR)
            cv2.imshow('Video', output_rgb)

            print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

        # indicate that the thread should be stopped
        self.stopped = True

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()


