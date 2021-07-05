import logging

from flloat.parser.ldlf import LDLfParser
from flloat.semantics import PLInterpretation
from load_and_save_data import Loader
from configuration import Config

#from gym_sapientino.core.types import Colors
from asyncio import sleep

from temprl.temprl.wrapper import TemporalGoal # --> Probabilmente dovrai sistemare il path --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

conf = Config()

class RestrainingBolt:

    def __init__(self, current_priorities):
        self._current_priorities = current_priorities



    offset = 1
    Colors = ['None', 'gold', 'orange', 'orange', 'brown']
    colors = list(map(str, Colors))[offset:]
    nb_colors = 2
    load = Loader()
    load.maps_data()
    #colors = load.all_priorities_points[offset:] # all_priorities_cells

    #print(colors, "col00000000000000000000000000000000000000000000000000")
    #nb_colors = len(colors)

    @classmethod
    def get_colors(cls):
        # Return only the colors of the actual priority on the map (then do not return all the colors): 
        #print("cls.colors[: cls.nb_colors]", cls.colors[: cls.nb_colors])
        #o = idx_pos()
        #offset = 1
        #Colors = Colorss[o]
        #colors = list(map(str, Colors))[offset:]
        #nb_colors = 2

        #return colors[: cls.nb_colors] # --> Poi togli questo '2'; se metti troppi colori sembra che non riesce a trovare una temporal goal (o per lo meno ci impiega molto tempo) --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        return cls.colors[: cls.nb_colors] #originale se si decommentano le variabili dove c'è nb_colors

    @classmethod
    def get_labels(cls):
        return cls.get_colors() + ["bad_beep"] # --> bad_beep --> ?????????????????????????


    @staticmethod
    def extract_sapientino_fluents(obs, action) -> PLInterpretation:
        fluents = set()
        if (conf.UNLIMITED_BATTERY == True):
            beep = obs[0]['beep']
            color_idx = int(obs[0]['color'])

        if color_idx == None:
            color_idx = 0
        if ((0 < color_idx <= len(RestrainingBolt.get_colors())) and beep):
            color_string = RestrainingBolt.get_colors()[color_idx - 1]
            fluents.add(color_string)

        elif ((color_idx == 0) and beep):
            fluents.add("bad_beep")
        result = PLInterpretation(fluents)

        return result

    @staticmethod
    def make_goal() -> str:
        """
        Define the goal for UAVs.
        :return: the string associated with the goal.
        """
        labels = RestrainingBolt.get_colors()
        # empty = "!bad_beep & !" + " & !".join(labels)
        empty = "!" + " & !".join(labels)
        f = "<(" + empty + ")*;{}>tt"
        regexp = (";(" + empty + ")*;").join(labels)
        f = f.format(regexp)
        #f= "<(!gold & !purple)*;gold;(!gold & !purple)*;purple>tt"
        print(f)

        return f

    @staticmethod
    def make_uavs_goal() -> TemporalGoal:
        s = RestrainingBolt.make_goal()
        logging.info(f"Computing {s}")
        return TemporalGoal(
            formula=LDLfParser()(s),
            reward=20.0,
            labels=set(RestrainingBolt.get_labels()),
            reward_shaping=True,
            zero_terminal_state=False,
            extract_fluents=RestrainingBolt.extract_sapientino_fluents,
        )

