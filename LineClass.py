# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        #x values of current fit
        self.current_x = None
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None