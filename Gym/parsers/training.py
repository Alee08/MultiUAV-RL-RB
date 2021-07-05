import sys
from configparser import ConfigParser

class TrainingConfig:

    def __init__(self, inifile='../settings/training.ini'):
        config = ConfigParser()
        l = config.read(inifile)

        if l==[]:
            print("ERROR cannot load settings init file")
            sys.exit(1)
        
        # [Training]

        self.section = config['Training']

        self.Q_LEARNING = False
        self.SARSA = False
        self.SARSA_lambda = False
        self.UAV_STANDARD_BEHAVIOUR = False
        self.algorithm = None
        self.R_MAX = False
        self.PRIOR_KNOWLEDGE = False

        if self.section['ALGORITHM']=='q_learning':
            self.Q_LEARNING = True
        elif self.section['ALGORITHM']=='baseline':
            self.UAV_STANDARD_BEHAVIOUR = False
        elif self.section['ALGORITHM']=='sarsa':
            self.SARSA = True
        elif self.section['ALGORITHM']=='sarsa_lambda':
            self.SARSA_lambda = True
        else:
            self.algorithm = 'invalid'
        self.EPISODES = int(self.section['EPISODES'])
        self.REDUCE_ITERATION_PER_EPISODE = True if self.section['REDUCE_ITERATION_PER_EPISODE']=='True' else False
        self.ITERATIONS_PER_EPISODE = int(self.section['EPISODES'])
        self.NO_INTERFERENCE_REWARD = True if self.section['EPISODES']=='True' else False
        self.EPISODES_BP = float(self.section['EPISODES'])
        self.Q_INIT_TYPE = self.section['Q_INIT_TYPE']

        if self.Q_INIT_TYPE=='max_reward':
            self.R_MAX = True
        elif self.Q_INIT_TYPE=='prior_knoledge':
            self.PRIOR_KNOWLEDGE = True
        elif self.Q_INIT_TYPE=='zero':
            pass
        else:
            self.Q_INIT_TYPE = 'invalid'

        self.validate()

    def validate(self):
        # Validating the values

        assert self.algorithm!='invalid', "No valid battery algorithm selection!"
        assert self.Q_INIT_TYPE!='invalid', "No valid Q-table initialization!"

    def inspect(self):
        print("============ ML config parameters ============")

        for k in self.__dict__.keys():
            print("%s: %r" %(k,self.__dict__[k]))

        print("-----------------------------------------------------------")

    def printkey(self,k):
        print("  %s = %r" %(k,self.__dict__[k]))

    def summary(self):
        print("============ Training parameters ============")

        self.printkey('algorithm')
        self.printkey('EPISODES')
        self.printkey('ITERATIONS_PER_EPISODE')
        self.printkey('REDUCE_ITERATION_PER_EPISODE')
        self.printkey('NO_INTERFERENCE_REWARD')
        self.printkey('EPISODES_BP')
        self.printkey('Q_INIT_TYPE')

        return ""