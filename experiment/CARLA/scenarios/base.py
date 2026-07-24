import carla



class BaseScenario:


    def __init__(
        self,
        world,
        external_control=True
    ):


        self.world = world


        # =====================
        # 控制权
        # =====================

        self.external_control = external_control

        # Set by the experiment runner to keep scenario clocks and CARLA ticks aligned.
        self.fixed_delta_s = 0.05



        # =====================
        # actor管理
        # =====================

        self.actors = []


        self.actor_roles = {}



        self.ego_vehicle = None


        self.npc_vehicles = []


        self.pedestrians = []


        self.obstacles = []



        # =====================
        # 场景状态
        # =====================

        self.status = "INITIALIZED"


        self.reason = ""


        self._finished = False




        # =====================
        # 场景信息
        # =====================

        self.scenario_id = "unknown"


        self.scenario_name = "unknown"



        # 路线

        self.route = []


        self.goal_location = None



        # 触发条件

        self.trigger = {}



        # 成功条件

        self.success_condition = {}



        # 失败条件

        self.failure_conditions = []





        # =====================
        # 指标
        # =====================

        self.metrics = {


            "collision_count": 0,


            "lane_invasion_count": 0,


            "simulation_time": 0,


            "min_distance": None,


            "reaction_time": None

        }






    """
    初始化场景
    """

    def setup(self):

        raise NotImplementedError






    """
    每帧更新环境

    external_control=True 时
    不允许控制Ego

    """

    def tick(self):

        pass






    """
    场景是否结束
    """

    def finished(self):

        return self._finished







    """
    场景成功
    """

    def success(
        self,
        reason="success"
    ):


        self.status = "SUCCESS"


        self.reason = reason


        self._finished = True








    """
    场景失败
    """

    def failure(
        self,
        reason="failure"
    ):


        self.status = "FAILURE"


        self.reason = reason


        self._finished = True







    """
    清理所有actor
    """

    def destroy(self):


        for actor in self.actors:


            if actor is not None:


                if actor.is_alive:


                    actor.destroy()



        self.actors.clear()


        self.actor_roles.clear()








    """
    注册actor


    role:

    ego

    front_vehicle

    walker

    obstacle

    """

    def add_actor(
        self,
        actor,
        role=None
    ):


        if actor is not None:


            self.actors.append(
                actor
            )


            if role is not None:


                self.actor_roles[role] = actor







    """
    获取actor id

    """

    def get_actor_ids(self):


        result = {}



        for role, actor in self.actor_roles.items():


            if actor is not None:


                result[role] = actor.id



        return result







    """
    对外提供场景信息

    """

    def get_scenario_info(self):


        return {


            "scenario_id":

            self.scenario_id,



            "scenario_name":

            self.scenario_name,



            "map_name":

            self.world.get_map().name,



            "external_control":

            self.external_control,



            "actors":

            self.get_actor_ids(),



            "route":

            self.route,



            "goal_location":

            self.goal_location,



            "trigger":

            self.trigger,



            "success_condition":

            self.success_condition,



            "failure_conditions":

            self.failure_conditions,



            "status":

            self.status,



            "reason":

            self.reason,



            "metrics":

            self.metrics

        }
