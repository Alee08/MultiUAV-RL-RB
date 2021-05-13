from temprl.temprl.wrapper import TemporalGoal, TemporalGoalWrapper
#from envs.custom_env_dir.custom_uav_env import UAVEnv


class UAVsTemporalWrapper(TemporalGoalWrapper):

    def step_agent(self, agent, action): # step_agent(self, agent, action)
        obs, reward, done, info = super().step_agent(agent, action) #super().step(agent, action)
        some_failed = any(tg.is_failed() for tg in self.temp_goals)
        all_true = all(tg.is_true() for tg in self.temp_goals)
        new_done = done or some_failed or all_true
        # if new_done:
        #     print(f"all_true={all_true}, some_failed={some_failed}")
        ##print(reward, "temp_wrapper")

        '''if (reward > 1):
            print(reward, "qiìuiii")
            breakpoint()
        else:
            print(reward, "quiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii")
        if reward > 1.0:
            breakpoint()'''
        return obs, reward, new_done, info
        # return obs, reward if reward > 0.0 else 0.0, new_done, info

