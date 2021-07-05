import sys
from configparser import ConfigParser

class ScenarioConfig:

    def __init__(self, inifile='../settings/scenario.ini'):
        config = ConfigParser()
        l = config.read(inifile)

        if l==[]:
            print("ERROR cannot load settings init file")
            sys.exit(1)
        
        # [Scenario]

        self.section = config['Scenario']

        self.HOSP_SCENARIO = True if self.section['SCENARIO_TYPE']=='hospitals' else False
        self.AREA_WIDTH = int(self.section['X'])
        self.AREA_HEIGHT = int(self.section['Y'])
        self.MAXIMUM_AREA_HEIGHT = int(self.section['Z'])
        self.CS_HEIGHT = int(self.section['CS_HEIGHT'])
        self.LOWER_BOUNDS = 0
        self.CELL_RESOLUTION_PER_COL = int(self.section['X_CELL_RESOLUTION'])
        self.CELL_RESOLUTION_PER_ROW = int(self.section['Y_CELL_RESOLUTION'])
        self.CREATE_ENODEB = True if self.section['CREATE_ENODEB']=='True' else False
        self.DIMENSION_2D = True if self.section['DIMENSION']=='2D' else False
        self.MIN_OBS_PER_AREA = float(self.section['MIN_OBS_PER_AREA'])
        self.MAX_OBS_PER_AREA = float(self.section['MAX_OBS_PER_AREA'])
        self.MIN_OBS_HEIGHT = float(self.section['MIN_OBS_HEIGHT'])
        self.MAX_OBS_HEIGHT = float(self.section['MAX_OBS_HEIGHT'])

        self.validate()

    def validate(self):
        # Validating the values

        assert self.MIN_OBS_PER_AREA>=0 and self.MAX_OBS_PER_AREA<=1, "No valid coverage percentage for obstacle generation!"
        assert self.MIN_OBS_HEIGHT>=0, "No valid minimum height per building!"
        assert self.CELL_RESOLUTION_PER_COL>=1 and self.CELL_RESOLUTION_PER_ROW>=1, "No valid coverage percentage for cell resolution!"
        assert self.section['DIMENSION']=='2D' or self.section['DIMENSION']=='3D', "No valid scenario dimension!"
        assert self.section['SCENARIO_TYPE']=='hospitals' or self.section['SCENARIO_TYPE']=='multi-service', "No valid type of scenario!"
        assert self.CS_HEIGHT<self.MAXIMUM_AREA_HEIGHT and self.CS_HEIGHT>=0, "No valid height for charging station!"

    def inspect(self):
        print("============ ML config parameters ============")

        for k in self.__dict__.keys():
            print("%s: %r" %(k,self.__dict__[k]))

        print("-----------------------------------------------------------")

    def printkey(self,k):
        print("  %s = %r" %(k,self.__dict__[k]))

    def summary(self):
        print("============ Scenario parameters ============")

        self.printkey('HOSP_SCENARIO')
        self.printkey('AREA_WIDTH')
        self.printkey('AREA_HEIGHT')
        self.printkey('MAXIMUM_AREA_HEIGHT')
        self.printkey('CELL_RESOLUTION_PER_COL')
        self.printkey('CELL_RESOLUTION_PER_COL')
        self.printkey('CREATE_ENODEB')
        self.printkey('DIMENSION_2D')
        self.printkey('CS_HEIGHT')
        self.printkey('MIN_OBS_PER_AREA')
        self.printkey('MAX_OBS_PER_AREA')
        self.printkey('MIN_OBS_HEIGHT')
        self.printkey('MAX_OBS_HEIGHT')

        return ""