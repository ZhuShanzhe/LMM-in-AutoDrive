class BaseScenario:


    def __init__(self, world):

        self.world = world

        self.actors = []

        self.ego_vehicle = None

        self.npc_vehicles = []

        self.pedestrians = []

        self.obstacles = []

        self._finished = False



    """
    初始化场景
    """

    def setup(self):

        raise NotImplementedError



    """
    每帧更新环境

    注意:
    不控制Ego
    """

    def tick(self):

        pass



    def finished(self):

        return self._finished



    """
    清理所有actor
    """

    def destroy(self):


        for actor in self.actors:


            if actor is not None:


                if actor.is_alive:

                    actor.destroy()



        self.actors.clear()



    """
    注册actor
    """

    def add_actor(
        self,
        actor
    ):


        if actor is not None:

            self.actors.append(
                actor
            )