class hosp_features():

    def __init__(self):

        self.P_GOLD = '#FFD700'
        self.P_ORANGE = "#F08000"

        self.P_BLU = "#0000FF"
        self.P_DARK_BLU = "#000080"

        self.P_GREEN = "#66ff99"
        self.P_DARK_GREEN = "#026b2c"

        self.P_FUCHSIA = "#FF00FF"
        self.P_PURPLE = "#cf03fc"

        self.P_PINK = "#FFB6C1"
        self.P_DARK_PINK= "#FF69B4"

        self.P_BROWN = "#A0522D"
        self.P_BROWN_DARK = "#8B4513"

        self.P_CYAN = "#7dbfc7"
        self.P_DARK_CYAN = "#79a3b6"

        self.P_LAVANDER = "#E6E6FA"
        self.P_DARK_LAVANDER = "#D8BFD8"

        self.P_NAVAJO= "#FFDEAD"
        self.P_DARK_NAVAJO = "#F4A460"

        self.P_GRAY = "#708090"
        self.P_DARK_GREY = "#2F4F4F"

        self.P_SALMON = "#FA8072"
        self.P_DARK_SALMON = "#E9967A"

        self.P_SKY_BLUE = "#ADD8E6"
        self.P_DARK_SKY_BLUE= "#00BFFF"

        self.P_LIGHT_GREEN = "#90EE90"
        self.P_DARK_LIGHT_GREEN = "#7FFF00"

        self.P_PAPAYA_YELLOW = "#FAFAD2"
        self.P_DARK_PAPAYA_YELLOW = "#FFEFD5"

        self.P_RED_LIGHT = "#DC143C"
        self.P_RED_LIGHT_DARK = "#B22222"

        self.P_RED = "#FF0000"
        self.P_DARK_RED = "#8B0000"

        self.HOSP_PRIORITIES = {1: self.P_GOLD, 2: self.P_ORANGE ,3: self.P_BLU, 4: self.P_DARK_BLU , 5: self.P_GREEN, 6: self.P_DARK_GREEN, 7: self.P_FUCHSIA, 8: self.P_PURPLE,
                                9: self.P_PINK, 10: self.P_DARK_PINK ,11: self.P_BROWN, 12: self.P_BROWN_DARK , 13: self.P_CYAN, 14: self.P_DARK_CYAN, 15: self.P_LAVANDER, 16: self.P_DARK_LAVANDER,
                                17: self.P_NAVAJO, 18: self.P_DARK_NAVAJO ,19: self.P_GRAY, 20: self.P_DARK_GREY , 21: self.P_SALMON, 22: self.P_DARK_SALMON, 23: self.P_SKY_BLUE, 24: self.P_DARK_SKY_BLUE,
                                25: self.P_LIGHT_GREEN, 26: self.P_DARK_LIGHT_GREEN ,27: self.P_PAPAYA_YELLOW, 28: self.P_DARK_PAPAYA_YELLOW , 29: self.P_RED_LIGHT, 30: self.P_RED_LIGHT_DARK}
    
        self.PRIORITY_NUM = len(self.HOSP_PRIORITIES)

    def get_color_name(self, color):

        if (color==self.P_GOLD):
            color_name = "gold"
        elif (color==self.P_ORANGE):
            color_name = "orange"
        elif (color == self.P_BLU):
            color_name = "blu"
        elif (color == self.P_DARK_BLU):
            color_name = "dark_blu"
        elif (color == self.P_GREEN):
            color_name = "green"
        elif (color == self.P_DARK_GREEN):
            color_name = "dark_green"
        elif (color == self.P_FUCHSIA):
            color_name = "fuchsia"
        elif (color==self.P_PURPLE):
            color_name = "purple"
        elif (color == self.P_PINK):
            color_name = "pink"
        elif (color==self.P_DARK_PINK):
            color_name = "dark_pink"
        elif (color == self.P_BROWN):
            color_name = "brown"
        elif (color == self.P_BROWN_DARK):
            color_name = "brown_dark"
        elif (color == self.P_CYAN):
            color_name = "cyan"
        elif (color == self.P_DARK_CYAN):
            color_name = "dark_cyan"
        elif (color == self.P_LAVANDER):
            color_name = "lavander"
        elif (color == self.P_DARK_LAVANDER):
            color_name = "dark_lavander"

        elif (color == P_RED):
            color_name = "red"
        else:
            color_name = None

        return color_name

    '''def get_color_id(self, color):
        color_id = list(self.HOSP_PRIORITIES.keys())[list(self.HOSP_PRIORITIES.values()).index(color)]

        return color_id'''

    def get_color_id(self, color):
        # print("GUARDA QUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA: ", color)

        if (color == self.P_GOLD):
            color_id = 1
        elif (color == self.P_ORANGE):
            color_id = 2
        elif (color == self.P_BLU):
            color_id = 3
        elif (color == self.P_DARK_BLU):
            color_id = 4
        elif (color == self.P_GREEN):
            color_id = 5
        elif (color == self.P_DARK_GREEN):
            color_id = 6
        elif (color == self.P_FUCHSIA):
            color_id = 7
        elif (color == self.P_PURPLE):
            color_id = 8
        elif (color == self.P_PINK):
            color_id = 9
        elif (color == self.P_DARK_PINK):
            color_id = 10
        elif (color == self.P_BROWN):
            color_id = 11
        elif (color == self.P_BROWN_DARK):
            color_id = 12
        elif (color == self.P_CYAN):
            color_id = 13
        elif (color == self.P_DARK_CYAN):
            color_id = 14
        elif (color == self.P_LAVANDER):
            color_id = 15
        elif (color == self.P_DARK_LAVANDER):
            color_id = 16
        else:
            color_id = 0

        return color_id