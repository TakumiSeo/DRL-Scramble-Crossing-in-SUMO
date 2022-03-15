import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import os

class Visualization:
    def __init__(self, path, dpi):
            self._path = path
            self._dpi = dpi


    def save_data_and_plot(self, data, filename, xlabel, ylabel):
        """
        Produce a plot of performance of the agent over the session and save the relative data to txt
        """
        min_val = min(data)
        max_val = max(data)

        plt.rcParams.update({'font.size': 24})  # set bigger font size

        plt.plot(data)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.margins(0)
        plt.ylim(min_val - 0.05 * abs(min_val), max_val + 0.05 * abs(max_val))
        fig = plt.gcf()
        fig.set_size_inches(20, 11.25)
        fig.savefig(os.path.join(self._path, 'plot_'+filename+'.png'), dpi=self._dpi)
        plt.close("all")

        with open(os.path.join(self._path, 'plot_'+filename + '_data.txt'), "w") as file:
            for value in data:
                    file.write("%s\n" % value)


    def std_calc(self, reward_list):
        v_list = []
        for i in range(len(reward_list)):
            if i <= 5:
                v_list.append(np.var(reward_list[i]))
            else:
                v_list.append(np.var(reward_list[i-5:i]))
        return [r - np.sqrt(v) for r, v in zip(list(reward_list), v_list)], [r + np.sqrt(v) for r, v in zip(list(reward_list), v_list)]

    def save_data_and_plotly_data(self, reward_data, reward_ped_data, x_rng, filename, multi=False):
        indices = [i for i in range(len(reward_data))]
        if not multi:
            p_all, m_all = self.std_calc(reward_list=reward_data)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=indices, y=p_all,
                                     fill=None,
                                     mode='lines',
                                     line_color='rgb(111,231,219)', name='stds plus', opacity=0.2))
            fig.add_trace(go.Scatter(x=indices, y=m_all, fill='tonexty', mode='lines', line_color='rgb(111,231,219)',
                                     name='stds minus', opacity=0.2))

            fig.add_trace(
                go.Scatter(x=indices, y=reward_data, mode='lines', line_color='blue', name='rewards', opacity=0.7))

            fig.update_layout(title="SUMO Intersection rewards", xaxis_title='episodes', yaxis_title='rewards',
                              legend_title='obtained values',
                              width=1000, height=600, font=dict(family='Courier New, monospace',
                                                                size=18,
                                                                color='RebeccaPurple'),
                              xaxis=dict(title='episodes', range=(-10, x_rng+10)),
                              yaxis=dict(title='reward', range=(-130000, 0)))
            fig.write_html(os.path.join(self._path, 'plot_' + filename + '.html'))
            fig.write_image(os.path.join(self._path, 'plot_' + filename + '.png'))
            plt.close("all")

            with open(os.path.join(self._path, 'plot_stdp_' + filename + '_data.txt'), "w") as file:
                for value in p_all:
                    file.write("%s\n" % value)

            with open(os.path.join(self._path, 'plot_stdm_' + filename + '_data.txt'), "w") as file:
                for value in m_all:
                    file.write("%s\n" % value)

            with open(os.path.join(self._path, 'plot_reward_' + filename + '_data.txt'), "w") as file:
                for value in reward_data:
                    file.write("%s\n" % value)

        else:
            m_v, p_v = self.std_calc(reward_data)
            m_p, p_p = self.std_calc(reward_ped_data)
            fig2 = go.Figure()

            fig2.add_trace(go.Scatter(x=indices, y=p_v,
                                      fill=None,
                                      mode='lines',
                                      line_color='rgb(183,168,204)', name='+stds', opacity=0.2))
            fig2.add_trace(go.Scatter(x=indices, y=m_v, fill='tonexty', mode='lines', line_color='rgb(183,168,204)',
                                      name='-stds', opacity=0.2))

            fig2.add_trace(
                go.Scatter(x=indices, y=reward_data, mode='lines', line_color='rgb(195,86,204)', name='vehicle rewards',
                           opacity=0.7))

            fig2.add_trace(go.Scatter(x=indices, y=p_p,
                                      fill=None,
                                      mode='lines',
                                      line_color='rgb(204,204,204)', name='+stds', opacity=0.2))
            fig2.add_trace(go.Scatter(x=indices, y=m_p, fill='tonexty', mode='lines', line_color='rgb(204,204,204)',
                                      name='-stds', opacity=0.2))

            fig2.add_trace(go.Scatter(x=indices, y=reward_ped_data, mode='lines', line_color='rgb(128,128,128)',
                                      name='pedestrian rewards', opacity=0.7))

            fig2.update_layout(title="SUMO Intersection Rewards", xaxis_title='episodes', yaxis_title='rewards',
                               legend_title='obtained values',
                               width=1000, height=600, font=dict(family='Courier New, monospace',
                                                                 size=18,
                                                                 color='RebeccaPurple'),
                               xaxis=dict(title='episodes', range=(-10, x_rng+10)),
                               yaxis=dict(title='reward', range=(-130000, 0)))
            fig2.write_html(os.path.join(self._path, 'plot_p_v_reward.html'))
            fig2.write_image(os.path.join(self._path, 'plot_p_v_reward.png'))
            plt.close("all")

            with open(os.path.join(self._path, 'plot_stdp_veh_data.txt'), "w") as file:
                for value in p_v:
                    file.write("%s\n" % value)

            with open(os.path.join(self._path, 'plot_stdm_veh_data.txt'), "w") as file:
                for value in m_v:
                    file.write("%s\n" % value)

            with open(os.path.join(self._path, 'plot_reward_veh_data.txt'), "w") as file:
                for value in reward_data:
                    file.write("%s\n" % value)


            with open(os.path.join(self._path, 'plot_stdp_ped_data.txt'), "w") as file:
                for value in p_p:
                    file.write("%s\n" % value)

            with open(os.path.join(self._path, 'plot_stdm_ped_data.txt'), "w") as file:
                for value in m_p:
                    file.write("%s\n" % value)

            with open(os.path.join(self._path, 'plot_reward_ped_data.txt'), "w") as file:
                for value in reward_ped_data:
                    file.write("%s\n" % value)
