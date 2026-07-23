import time
import json


from Route_manager import RouteManager



class ScenarioManager:


    def __init__(
        self,
        world,
        fixed_delta_s=0.05
    ):


        self.world = world

        self.fixed_delta_s = fixed_delta_s


        # 注册表

        self.scenarios = {}



        # 单场景

        self.current = None



        # 大场景

        self.route_manager = (
            RouteManager(world)
        )


        self.events = []

        self.active = []




    # =====================
    # 注册
    # =====================

    def register(
        self,
        name,
        cls
    ):

        self.scenarios[name]=cls




    # =====================
    # 单场景
    # =====================

    def load(
        self,
        name,
        external_control=True
    ):


        scenario = self.scenarios[name](
            self.world,
            external_control
        )


        scenario.fixed_delta_s = (
            self.fixed_delta_s
        )


        self.current=scenario


        return scenario




    def setup(self):

        if self.current:

            self.current.setup()




    # =====================
    # 加载8km配置
    # =====================

    def load_long_scene(
        self,
        config_path
    ):


        with open(
            config_path,
            encoding="utf8"
        ) as f:

            cfg=json.load(f)



        route_cfg=cfg["route"]



        self.route_manager.build_route(
            length=route_cfg["length"],
            step=route_cfg["step"]
        )



        self.events=cfg["events"]



        print(
            "[LongScene] Loaded events:",
            len(self.events)
        )





    # =====================
    # 大场景tick
    # =====================

    def tick_long_scene(
        self,
        ego_vehicle
    ):


        self.route_manager.update(
            ego_vehicle
        )


        progress=(
            self.route_manager
            .get_progress()
        )



        # 触发事件

        for event in self.events:


            if event.get(
                "triggered",
                False
            ):

                continue



            if progress >= event["distance"]:


                self.spawn_event(
                    event["scenario"]
                )


                event["triggered"]=True





        # 更新事件


        for scenario in self.active[:]:


            scenario.tick()



            if scenario.finished():


                print(
                    "[LongScene] finish",
                    scenario.get_status()
                )


                scenario.destroy()


                self.active.remove(
                    scenario
                )





    # =====================
    # 创建事件
    # =====================

    def spawn_event(
        self,
        name
    ):


        if name not in self.scenarios:

            print(
                "[Warning] missing scenario:",
                name
            )

            return



        scenario=self.scenarios[name](
            self.world,
            True
        )


        scenario.fixed_delta_s=(
            self.fixed_delta_s
        )


        scenario.setup()



        self.active.append(
            scenario
        )



        print(
            "[LongScene] Spawn:",
            name
        )





    # =====================
    # 单场景run
    # =====================

    def run(self):


        while not self.current.finished():


            self.current.tick()


            self.world.tick()


            time.sleep(
                self.fixed_delta_s
            )




    def destroy(self):


        if self.current:

            self.current.destroy()


        for s in self.active:

            s.destroy()


        self.active.clear()