import threading
import cv2
import time


class RealTimeVideoCapture:
    """
    VideoCapture
    ------------
    This class is designed to be a wrapper for cv2.VideoCapture to streamline
    video feed handling by using threading.

    The primary function is that the read() function is run constantly in a separate thread.
    Locks and Events are used to allow this class to be thread-safe.


    Parameters:
        video_source (str): Path to a video file or a video feed URL (rtsp/rtmp/http).
        capped_fps (bool): If True, caps the frame rate (default is False). Set to true for file playback. 
        framerate (int): Frame rate for video file playback (used if capped_fps is True).
        restart_on_end (bool): If True, restarts video file playback when it ends (default is False).
    """
    
    last_frame = None
    last_ready = False
    lock = threading.Lock()
    stop_event = threading.Event()
    start_event = threading.Event()
    fps = 30
    video_source = None
    capped_fps = False
    restart_on_end = False

    # make sure capped_fps is False for case of rstp/rtmp url 
    def __init__(self, video_source:str, framerate:int=30, capped_fps:bool=False, restart_on_end:bool=False):
        self.fps : int = framerate
        self.video_source : str = video_source
        self.capped_fps : bool = capped_fps
        self.restart_on_end : bool = restart_on_end
        self.cap : cv2.VideoCapture = cv2.VideoCapture(video_source)
        self.thread : threading.Thread = threading.Thread(target=self.__capture_read_thread__)
        self.thread.daemon = True
        self.thread.start()

    def __capture_read_thread__(self):
        """
        Continuously reads frames from the video source in a separate thread.

        This method is intended to run in a separate thread and is not meant to be called directly.
        It reads frames as soon as they are available, and handles video restart if specified.
        If capped_fps is True, it waits to maintain the specified frame rate.

        The method stops when stop_event is set or if the video source cannot provide frames and restart_on_end is False.
        """
        while not self.stop_event.is_set():
            if self.start_event.is_set():
                self.last_ready, last_frame = self.cap.read()
                if self.last_ready:
                    with self.lock:
                        self.last_frame = last_frame
                else:
                    self.last_frame = None

                # print(self.video_source, " frame read: ", self.last_ready)
            
                if not self.last_ready and self.restart_on_end:  # restart if file playback video ended
                    with self.lock:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
                # only wait in case of video files, else keep reading frames without delay to prevent packet drop/burst receive
                if self.capped_fps:
                    time.sleep(1 / self.fps)
        return
    
    def start(self):
        """
        Start the video capture reading frames.
        """
        print("Capture thread has started.")
        self.start_event.set()
        # sleep to allow
        time.sleep(1/self.fps) # sleep for a tiny bit to allow capture thread to start properly

    def read(self):
        """
        Retrieve the latest frame from the video capture. Skips frames automatically 
        if they are not read in time. Allows for realtime video playback.

        Returns:
            [success boolean, frame or None]
        """
        try:
            if self.start_event.is_set():
                # with self.lock:
                if (self.last_ready is not False) and (self.last_frame is not None):
                    return [self.last_ready, self.last_frame.copy()]
            else:
                if not self.stop_event.is_set():
                    self.start()

            return [False, None]
        except Exception as e:
            raise ValueError(f"Error encountered by read() function: {e}")
        
    def isOpened(self):
        """
        Check if the video source is opened.
        """
        return self.cap.isOpened()    

    def open(self, video_source:str=None):
        """
        Open a new video source.
        
        Args:
            video_source (str): Path to the new video source.
        """
        if video_source is not None:
            self.video_source = video_source
        
        with self.lock:
            self.cap.open(self.video_source)

    def release(self):
        """
        Stop the video capture and release resources.
        """
        self.stop_event.set()
        self.restart_on_end = False
        self.thread.join(2)
        
        with self.lock:
            self.cap.release()
