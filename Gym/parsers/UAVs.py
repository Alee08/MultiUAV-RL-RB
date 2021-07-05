import sys
from configparser import ConfigParser

class UAVsConfig:

    def __init__(self, inifile='../settings/UAVs.ini'):
        config = ConfigParser()
        l = config.read(inifile)

        if l==[]:
            print("ERROR cannot load settings init file")
            sys.exit(1)
        
        # [UAVs]

        self.section = config['UAVs']

        self.UNLIMITED_BATTERY = True if self.section['BATTERY_AUTONOMY_TIME']=='None' else False
        if isinstance(self.section['BATTERY_AUTONOMY_TIME'], int):
            self.BATTERY_AUTONOMY_TIME = int(self.section['BATTERY_AUTONOMY_TIME'])
        else:
            if self.section['BATTERY_AUTONOMY_TIME']=='None':
                self.BATTERY_AUTONOMY_TIME = None
            else:
                self.BATTERY_AUTONOMY_TIME = 'invalid'
        self.MIN_UAV_Z = int(self.section['MIN_UAV_Z'])
        self.NOISE_ON_POS_MEASURE = True if self.section['NOISE_ON_POS_MEASURE']=='True' else False
        self.MULTI_SERVICE = self.section['MULTI_SERVICE']
        self.UAV_BANDWIDTH = float(self.section['UAV_BANDWIDTH'])
        self.UAV_FOOTPRINT = float(self.section['UAV_FOOTPRINT'])
        self.TIME_SLOT_FOR_DELAYED_START = int(self.section['TIME_SLOT_FOR_DELAYED_START'])

        self.validate()

    def validate(self):
        # Validating the values

        assert self.UNLIMITED_BATTERY!='inavlid', "No valid battery autonomy value!"
        assert self.MIN_UAV_Z>=0, "No valid minimum flight height per UAV!"
        assert self.UAV_FOOTPRINT>=0, "No valid UAV footprint!"
        assert self.TIME_SLOT_FOR_DELAYED_START>=0, "No valid time for delyed take-off!"

    def inspect(self):
        print("============ ML config parameters ============")

        for k in self.__dict__.keys():
            print("%s: %r" %(k,self.__dict__[k]))

        print("-----------------------------------------------------------")

    def printkey(self,k):
        print("  %s = %r" %(k,self.__dict__[k]))

    def summary(self):
        print("============ UAVs parameters ============")

        self.printkey('UNLIMITED_BATTERY')
        self.printkey('BATTERY_AUTONOMY_TIME')
        self.printkey('MIN_UAV_Z')
        self.printkey('NOISE_ON_POS_MEASURE')
        self.printkey('MULTI_SERVICE')
        self.printkey('UAV_BANDWIDTH')
        self.printkey('UAV_FOOTPRINT')
        self.printkey('TIME_SLOT_FOR_DELAYED_START')

        return ""