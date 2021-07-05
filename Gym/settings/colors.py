class Colors():

    def __init__(self):

        self.P_BLU = "#0000FF"

        self.P_DARK_GREEN = "#026b2c"

        self.P_FUCHSIA = "#FF00FF"

        self.P_DARK_PINK= "#FF69B4"

        self.P_CYAN = "#7dbfc7"

        self.P_PINK = "#FFB6C1"

        self.P_DARK_LAVANDER = "#D8BFD8"

        self.P_SALMON = "#FA8072"

        self.P_DARK_SKY_BLUE= "#00BFFF"

        self.P_DARK_PAPAYA_YELLOW = "#FFEFD5"

        self.P_RED_LIGHT = "#DC143C"

        self.P_RED = "#FF0000"
        self.P_DARK_RED = "#8B0000"

        self.WHITE = "#ffffff"
        self.LIGHT_RED = "#ff0000"
        self.DARK_RED = "#800000"
        self.LIGHT_BLUE = "#66ffff"
        self.DARK_BLUE = "#000099"
        self.LIGHT_GREEN = "#66ff99"
        self.DARK_GREEN = "#006600"
        self.PURPLE = "#cf03fc"
        self.ORANGE = "#fc8c03"
        self.BROWN = "#8b4513"
        self.GOLD = '#FFD700'
        self.HEX_CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'] # Used to generate random hex color related to the users clusters.
        self.HEX_COLOR_CODE = '#'
        
        # UAVS colors:
        self.VIOLET = '#EE82EE'
        self.ORANGE = '#FFA500'
        self.GREY = '#808080'
        self.BROWN = '#A52A2A'

        self.UAVS_COLORS = [self.P_RED_LIGHT, self.ORANGE, self.P_BLU, self.P_DARK_GREEN, self.VIOLET, self.BROWN, self.GREY, self.P_SALMON, self.P_DARK_LAVANDER, self.P_CYAN, self.P_DARK_PAPAYA_YELLOW, self.P_PINK, self.P_DARK_SKY_BLUE, self.P_DARK_PINK, self.P_FUCHSIA]
        
        self.UAVS_COLORS_RGB_PERCENTAGE = [Colors.hex_to_rgb(color) for color in self.UAVS_COLORS]

    @staticmethod
    def hex_to_rgb(hex):
        hex = hex.lstrip('#')
        hlen = len(hex)
        rgb = tuple(int(hex[i:i+hlen//3], 16) for i in range(0, hlen, hlen//3))
        rgb_percentage = [color_part/255 for color_part in rgb]
        
        return tuple(rgb_percentage)