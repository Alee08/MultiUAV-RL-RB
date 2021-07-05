import sys
from configparser import ConfigParser

class UsersConfig:

    def __init__(self, inifile='../settings/users.ini'):
        config = ConfigParser()
        l = config.read(inifile)

        if l==[]:
            print("ERROR cannot load settings init file")
            sys.exit(1)
        
        # [users]

        self.section = config['Users']

        self.MAX_USER_Z = int(self.section['MAX_USER_Z'])
        self.INF_REQUEST = True if self.section['INF_REQUEST']=='True' else False
        self.STATIC_REQUEST = True if self.section['STATIC_REQUEST']=='True' else False
        self.USERS_PRIORITY = True if self.section['USERS_PRIORITY']=='True' else False
        self.MOVE_USERS_EACH_N_EPOCHS = int(self.section['MOVE_USERS_EACH_N_EPOCHS'])
        self.UPDATE_USERS_REQUESTS_EACH_N_ITERATIONS = int(self.section['UPDATE_USERS_REQUESTS_EACH_N_ITERATIONS'])

        self.validate()

    def validate(self):
        # Validating the values

        pass

    def inspect(self):
        print("============ ML config parameters ============")

        for k in self.__dict__.keys():
            print("%s: %r" %(k,self.__dict__[k]))

        print("-----------------------------------------------------------")

    def printkey(self,k):
        print("  %s = %r" %(k,self.__dict__[k]))

    def summary(self):
        print("============ Users parameters ============")

        self.printkey('MAX_USER_Z')
        self.printkey('INF_REQUEST')
        self.printkey('STATIC_REQUEST')
        self.printkey('USERS_PRIORITY')
        self.printkey('MOVE_USERS_EACH_N_EPOCHS')
        self.printkey('UPDATE_USERS_REQUESTS_EACH_N_ITERATIONS')

        return ""