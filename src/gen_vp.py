import numpy as np
import math

class TrafficGenerator:
    def __init__(self, max_steps, n_cars_generated, n_peds_generated):
        self._n_cars_generated = n_cars_generated  # how many cars per episode
        self._n_peds_generated = n_peds_generated
        self._max_steps = max_steps

    def generate_routefile(self, seed):
        """
        Generation of the route of every car for one episode
        """
        np.random.seed(seed)  # make tests reproducible

        # the generation of cars is distributed according to a weibull distribution
        timings = np.random.weibull(2, self._n_cars_generated)
        ped_timings = np.random.uniform(2, self._n_peds_generated, self._n_peds_generated)
        timings = np.sort(timings)
        ped_timings = np.sort(ped_timings)

        # reshape the distribution to fit the interval 0:max_steps
        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        min_new = 0
        max_new = self._max_steps
        for value in timings:
            car_gen_steps = np.append(car_gen_steps, ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)

        car_gen_steps = np.rint(car_gen_steps)  # round every value to int -> effective steps when a car will be generated

        # apply same method to pedestrian
        ped_gen_steps = []
        min_old_ped = math.floor(ped_timings[1])
        max_old_ped = math.ceil(ped_timings[-1])
        min_new_ped = 0
        max_new_ped = self._max_steps -1400

        for val in ped_timings:
            ped_gen_steps = np.append(ped_gen_steps,  ((max_new_ped - min_new_ped) / (max_old_ped - min_old_ped)) * (val - max_old_ped) + max_new_ped)

        ped_gen_steps = np.rint(ped_gen_steps)
        ped_gen_steps[0] = 0
        # produce the file for cars generation, one car per line
        with open("intersection/episode_routes.rou.xml", "w") as routes:
            print('<?xml version="1.0" encoding="UTF-8"?>', file=routes)
            print("""<routes>
            <vType id="slow" length="5" maxSpeed="8.33" accel="2.6" decel="4.5" speedDev="0.5" sigma="0.2" vClass="passenger"/>
            <vType id="medium" length="5" maxSpeed="13.9" accel="2.6" decel="4.5" speedDev="0.5" sigma="0.2" vClass="passenger"/>
            <vType id="fast" length="5" maxSpeed="22.2" accel="2.6" decel="4.5" speedDev="0.5" sigma="0.2" vClass="passenger"/>
            <route id="W_E" edges="WW2TL W2TL TL2E TL2EE"/>
            <route id="E_W" edges="EE2TL E2TL TL2W TL2WW"/>
            <route id="N_S" edges="NN2TL N2TL TL2S TL2SS"/>
            <route id="S_N" edges="SS2TL S2TL TL2N TL2NN"/>
        
            <route id="S_W" edges="SS2TL S2TL TL2W TL2WW"/>
            <route id="W_N" edges="WW2TL W2TL TL2N TL2NN"/>
            <route id="N_E" edges="NN2TL N2TL TL2E TL2EE"/>
            <route id="E_S" edges="EE2TL E2TL TL2S TL2SS"/>
        
            <route id="S_E" edges="SS2TL S2TL TL2E TL2EE"/>
            <route id="W_S" edges="WW2TL W2TL TL2S TL2SS"/>
            <route id="N_W" edges="NN2TL N2TL TL2W TL2WW"/>
            <route id="E_N" edges="EE2TL E2TL TL2N TL2NN"/>""", file=routes)

            print("""
            <flow id="WE" route="W_E" type="medium" begin="0" end="5400" probability="0.05" departLane="2" departSpeed="5" /> 
            <flow id="EW" route="E_W" type="medium" begin="0" end="5400" probability="0.05" departLane="2" departSpeed="5" />
            <flow id="NS" route="N_S" type="medium" begin="0" end="5400" probability="0.02" departLane="2" departSpeed="5" />
            <flow id="SN" route="S_N" type="medium" begin="0" end="5400" probability="0.05" departLane="2" departSpeed="5" />

            <flow id="SW" route="S_W" type="slow" begin="0" end="5400" probability="0.05" departLane="1" departSpeed="5" />
            <flow id="WN" route="W_N" type="slow" begin="0" end="5400" probability="0.05" departLane="1" departSpeed="5" />
            <flow id="NE" route="N_E" type="slow" begin="0" end="5400" probability="0.02" departLane="1" departSpeed="5" />
            <flow id="ES" route="E_S" type="slow" begin="0" end="5400" probability="0.05" departLane="1" departSpeed="5" />

            <flow id="SE" route="S_E" type="medium" begin="0" end="5400" probability="0.05" departLane="random" departSpeed="5" />
            <flow id="WS" route="W_S" type="medium" begin="0" end="5400" probability="0.05" departLane="random" departSpeed="5" />
            <flow id="NW" route="N_W" type="fast" begin="0" end="5400" probability="0.02" departLane="random" departSpeed="5" />
            <flow id="EN" route="E_N" type="fast" begin="0" end="5400" probability="0.05" departLane="random" departSpeed="5" />
            """, file=routes)

            for ped_counter, p_step in enumerate(ped_gen_steps):
                diag_ped = np.random.randint(0, 100)
                if diag_ped <= 72:
                    if p_step <= 0:
                        p_step = 0.0
                    ped_straight = np.random.randint(1, 9)
                    # S2TL TL2N
                    if ped_straight == 1:
                        print('    <vType id="p%i" vClass="pedestrian" width="0.5" length="0.21" minGap="0.2" maxSpeed="1.5" guiShape="pedestrian"/>\n' % ped_counter, file=routes)
                        print('    <person id="p%i" type="p%i" depart="%s" departPos="0">\n' % (ped_counter, ped_counter, p_step), file=routes)
                        print('        <walk from="S2TL" to="TL2N" arrivalPos="-1"/>\n', file=routes)
                        print('    </person>\n', file=routes)
                    elif ped_straight == 2:
                        print('    <vType id="p%i" vClass="pedestrian" width="0.5" length="0.21" minGap="0.2" maxSpeed="1.5" guiShape="pedestrian"/>\n' % ped_counter, file=routes)
                        print('    <person id="p%i" type="p%i" depart="%s" departPos="0">\n' % (ped_counter, ped_counter, p_step), file=routes)
                        print('        <walk from="TL2N" to="S2TL" arrivalPos="-1"/>\n', file=routes)
                        print('    </person>\n', file=routes)
                    # TL2S N2TL
                    elif ped_straight == 3:
                        print('    <vType id="p%i" vClass="pedestrian" width="0.5" length="0.21" minGap="0.2" maxSpeed="1.5" guiShape="pedestrian"/>\n' % ped_counter, file=routes)
                        print('    <person id="p%i" type="p%i" depart="%s" departPos="0">\n' % (ped_counter, ped_counter, p_step), file=routes)
                        print('        <walk from="TL2S" to="N2TL" arrivalPos="-1"/>\n', file=routes)
                        print('    </person>\n', file=routes)
                    elif ped_straight == 4:
                        print('    <vType id="p%i" vClass="pedestrian" width="0.5" length="0.21" minGap="0.2" maxSpeed="1.5" guiShape="pedestrian"/>\n' % ped_counter, file=routes)
                        print('    <person id="p%i" type="p%i" depart="%s" departPos="0">\n' % (ped_counter, ped_counter, p_step), file=routes)
                        print('        <walk from="N2TL" to="TL2S" arrivalPos="-1"/>\n', file=routes)
                        print('    </person>\n', file=routes)
                    # E2TL W2TL
                    elif ped_straight == 5:
                        print('    <vType id="p%i" vClass="pedestrian" width="0.5" length="0.21" minGap="0.2" maxSpeed="1.5" guiShape="pedestrian"/>\n' % ped_counter, file=routes)
                        print('    <person id="p%i" type="p%i" depart="%s" departPos="0">\n' % (ped_counter, ped_counter, p_step), file=routes)
                        print('        <walk from="E2TL" to="W2TL" arrivalPos="-1"/>\n', file=routes)
                        print('    </person>\n', file=routes)
                    elif ped_straight == 6:
                        print('    <vType id="p%i" vClass="pedestrian" width="0.5" length="0.21" minGap="0.2" maxSpeed="1.5" guiShape="pedestrian"/>\n' % ped_counter, file=routes)
                        print('    <person id="p%i" type="p%i" depart="%s" departPos="0">\n' % (ped_counter, ped_counter, p_step), file=routes)
                        print('        <walk from="W2TL" to="E2TL" arrivalPos="-1"/>\n', file=routes)
                        print('    </person>\n', file=routes)
                    # TL2W TL2E
                    elif ped_straight == 7:
                        print('    <vType id="p%i" vClass="pedestrian" width="0.5" length="0.21" minGap="0.2" maxSpeed="1.5" guiShape="pedestrian"/>\n' % ped_counter, file=routes)
                        print('    <person id="p%i" type="p%i" depart="%s" departPos="0">\n' % (ped_counter, ped_counter, p_step), file=routes)
                        print('        <walk from="TL2W" to="TL2E" arrivalPos="-1"/>\n', file=routes)
                        print('    </person>\n', file=routes)
                    elif ped_straight == 8:
                        print('    <vType id="p%i" vClass="pedestrian" width="0.5" length="0.21" minGap="0.2" maxSpeed="1.5" guiShape="pedestrian"/>\n' % ped_counter, file=routes)
                        print('    <person id="p%i" type="p%i" depart="%s" departPos="0">\n' % (ped_counter, ped_counter, p_step), file=routes)
                        print('        <walk from="TL2E" to="TL2W" arrivalPos="-1"/>\n', file=routes)
                        print('    </person>\n', file=routes)
                else:
                    ped_diag = np.random.randint(1, 9)
                    # x crosswalk
                    # right 1-1
                    if ped_diag == 1:
                        print('    <vType id="p%i" vClass="pedestrian" width="0.5" length="0.21" minGap="0.2" maxSpeed="1.5" guiShape="pedestrian"/>\n' % ped_counter, file=routes)
                        print('    <person id="p%i" type="p%i" depart="%s" departPos="0">\n' % (ped_counter, ped_counter, p_step), file=routes)
                        print('        <walk from="N2TL" to="TL2W" arrivalPos="-1"/>\n', file=routes)
                        print('    </person>\n', file=routes)
                    elif ped_diag == 2:
                        print('    <vType id="p%i" vClass="pedestrian" width="0.5" length="0.21" minGap="0.2" maxSpeed="1.5" guiShape="pedestrian"/>\n' % ped_counter, file=routes)
                        print('    <person id="p%i" type="p%i" depart="%s" departPos="0">\n' % (ped_counter, ped_counter, p_step), file=routes)
                        print('        <walk from="TL2W" to="N2TL" arrivalPos="-1"/>\n', file=routes)
                        print('    </person>\n', file=routes)
                    # right 1-2
                    elif ped_diag == 3:
                        print('    <vType id="p%i" vClass="pedestrian" width="0.5" length="0.21" minGap="0.2" maxSpeed="1.5" guiShape="pedestrian"/>\n' % ped_counter, file=routes)
                        print('    <person id="p%i" type="p%i" depart="%s" departPos="0">\n' % (ped_counter, ped_counter, p_step), file=routes)
                        print('        <walk from="E2TL" to="S2TL" arrivalPos="-1"/>\n', file=routes)
                        print('    </person>\n', file=routes)
                    elif ped_diag == 4:
                        print('    <vType id="p%i" vClass="pedestrian" width="0.5" length="0.21" minGap="0.2" maxSpeed="1.5" guiShape="pedestrian"/>\n' % ped_counter, file=routes)
                        print('    <person id="p%i" type="p%i" depart="%s" departPos="0">\n' % (ped_counter, ped_counter, p_step), file=routes)
                        print('        <walk from="S2TL" to="E2TL" arrivalPos="-1"/>\n', file=routes)
                        print('    </person>\n', file=routes)

                    # left 1-1
                    elif ped_diag == 5:
                        print('    <vType id="p%i" vClass="pedestrian" width="0.5" length="0.21" minGap="0.2" maxSpeed="1.5" guiShape="pedestrian"/>\n' % ped_counter, file=routes)
                        print('    <person id="p%i" type="p%i" depart="%s" departPos="0">\n' % (ped_counter, ped_counter, p_step), file=routes)
                        print('        <walk from="TL2N" to="TL2E" arrivalPos="-1"/>\n', file=routes)
                        print('    </person>\n', file=routes)
                    elif ped_diag == 6:
                        print('    <vType id="p%i" vClass="pedestrian" width="0.5" length="0.21" minGap="0.2" maxSpeed="1.5" guiShape="pedestrian"/>\n' % ped_counter, file=routes)
                        print('    <person id="p%i" type="p%i" depart="%s" departPos="0">\n' % (ped_counter, ped_counter, p_step), file=routes)
                        print('        <walk from="TL2E" to="TL2N" arrivalPos="-1"/>\n', file=routes)
                        print('    </person>\n', file=routes)
                    # left 1-2
                    elif ped_diag == 7:
                        print('    <vType id="p%i" vClass="pedestrian" width="0.5" length="0.21" minGap="0.2" maxSpeed="1.5" guiShape="pedestrian"/>\n' % ped_counter, file=routes)
                        print('    <person id="p%i" type="p%i" depart="%s" departPos="0">\n' % (ped_counter, ped_counter, p_step), file=routes)
                        print('        <walk from="TL2N" to="TL2S" arrivalPos="-1"/>\n', file=routes)
                        print('    </person>\n', file=routes)
                    elif ped_diag == 8:
                        print('    <vType id="p%i" vClass="pedestrian" width="0.5" length="0.21" minGap="0.2" maxSpeed="1.5" guiShape="pedestrian"/>\n' % ped_counter, file=routes)
                        print('    <person id="p%i" type="p%i" depart="%s" departPos="0">\n' % (ped_counter, ped_counter, p_step), file=routes)
                        print('        <walk from="TL2S" to="W2TL" arrivalPos="-1"/>\n', file=routes)
                        print('    </person>\n', file=routes)

            print("</routes>", file=routes)

gen = TrafficGenerator(max_steps=5400, n_cars_generated=1000, n_peds_generated=300)
gen.generate_routefile(seed=100)