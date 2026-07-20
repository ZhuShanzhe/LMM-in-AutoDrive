import carla
import time



class ScenarioRunner:


    def __init__(
        self,
        host="localhost",
        port=2000
    ):


        self.client = carla.Client(
            host,
            port
        )


        self.client.set_timeout(
            10.0
        )


        self.world = (
            self.client
            .get_world()
        )


        self.scenario = None





    def load_scenario(
        self,
        scenario
    ):


        self.scenario = scenario


        print(
            "[Runner] Scenario loaded"
        )






    def run(self):


        if self.scenario is None:


            raise RuntimeError(
                "No scenario loaded"
            )



        print(
            "[Runner] Setup..."
        )



        self.scenario.setup()



        print(
            "[Runner] Running..."
        )



        try:


            while not self.scenario.finished():


                self.scenario.tick()


                time.sleep(
                    0.05
                )



        finally:


            print(
                "[Runner] Result:"
            )


            print(
                self.scenario.get_scenario_info()
            )



            print(
                "[Runner] Cleaning..."
            )


            self.scenario.destroy()



            print(
                "[Runner] Finished"
            )