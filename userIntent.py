class UserIntent():
    def __init__(self):
        self.click_x = None
        self.click_y = None
        self.auto_acquire = False
        self.ROI_coords = None
        self.runML = False

    def set_click_coordinates(self, x, y):
        '''
        Set the coordinates where the user clicked on the video feed.
        '''
        self.click_x = int(y * 640) - 50  # adjust for canvas offset
        self.click_y = int(x * 640)
        print(f"Set click coordinates to: ({self.click_x}, {self.click_y})")
    
    def set_ROI(self, coords):
        '''
        coords is [x, y, w, h]
        '''
        self.ROI_coords = [int(c * 640) for c in coords]
        print(f"Set ROI to: {self.ROI_coords}")
    
    def set_ML(self, status: bool = None):
        if status is None:
            # toggle
            self.runML = not self.runML
        else:
            self.runML = status
    
    def clear_click_coordinates(self):
        self.click_x = None
        self.click_y = None
        print("cleared click coordinates")
    
    def clear_ROI(self):
        self.ROI_coords = None
        print("cleared ROI")