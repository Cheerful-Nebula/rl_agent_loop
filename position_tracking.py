import numpy as np
import matplotlib.pyplot as plt
import os
import platform
from datetime import date


class PositionTracker():

    def __init__(self, obs_idxs: list, expt_num: int, learning_algo: str, env_id: str):
        self.obs_idxs = obs_idxs
        self.pos_plotting_pts = {i: {'timesteps': [], 'avg': [], 'max': [], 'min': []} for i in self.obs_idxs}
        self.term_plotting_cnt = []
        self.trun_plotting_cnt = []
        self.expt_num = expt_num
        self.learning_algo = learning_algo
        self.env_id = env_id
        self.today = date.today()

    def start_rollout(self):
        self.rollout_term_cnt = 0
        self.rollout_trun_cnt = 0
        self.rollout_pos_pts = {i: {'avg': [], 'max': [], 'min': []} for i in self.obs_idxs}

    def start_episode(self):
        self.ep_pos_pts = {i: [] for i in self.obs_idxs}

    def track_episode(self, obs):
        for i in self.obs_idxs:
            self.ep_pos_pts[i].append(obs[i])

    def end_episode(self, terminated):
        if terminated:
            self.rollout_term_cnt += 1
        else:
            self.rollout_trun_cnt += 1

        # converting list of position points to numpy array
        for i in self.obs_idxs:
            self.ep_pos_pts[i] = np.array(self.ep_pos_pts[i])

        # aggergating position points from episode
        self.ep_pts_agg = {i: {'avg': float(np.mean(self.ep_pos_pts[i])),
                               'max': float(np.max(self.ep_pos_pts[i])),
                               'min': float(np.min(self.ep_pos_pts[i]))} for i in self.obs_idxs}

        # loop through each -avg,-max,-min dict of each obersavtion index for this episode
        # append number to dictionary containing values position values for rollout
        for i in self.obs_idxs:
            for j in self.ep_pts_agg[i].keys():
                self.rollout_pos_pts[i][j].append(self.ep_pts_agg[i][j])

    def end_rollout(self, num_timesteps: int = None):
        # converting list of position points to numpy arrays
        for i in self.obs_idxs:
            for j in ['avg', 'max', 'min']:
                self.rollout_pos_pts[i][j] = np.array(self.rollout_pos_pts[i][j])

        # average the averages across all episodes in rollout
        self.rollout_pts_agg = {i: {'timesteps': num_timesteps,
                                    'avg': float(np.mean(self.rollout_pos_pts[i].get('avg'))),
                                    'max': float(np.max(self.rollout_pos_pts[i].get('max'))),
                                    'min': float(np.min(self.rollout_pos_pts[i].get('min')))} for i in self.obs_idxs}

        for i in self.obs_idxs:
            for j in self.rollout_pts_agg[i].keys():
                self.pos_plotting_pts[i][j].append(self.rollout_pts_agg[i][j])

        self.term_plotting_cnt.append(self.rollout_term_cnt)
        self.trun_plotting_cnt.append(self.rollout_trun_cnt)

    def plot_position(self, title: str = None):
        """
        Plot position tracking data for each observation index.

        This method creates line plots showing the average, maximum, and minimum
        position values across timesteps for each observation index. Each plot
        includes markers, a title, legend, axis labels, and a logging note.

        Args:
            title (str, optional): Custom title for the plot. If not provided,
                a default title using the observation index will be used.

        Side effects:
            - Populates self.pos_fig_list with tuples of (observation_index, figure)
            - Creates matplotlib figures for each observation index
        """

        self.pos_fig_list = []
        logging_note = f'{self.today} | Experiment No. {self.expt_num} | Learning Algo: {self.learning_algo} | Environment ID: {self.env_id} |'

        for i in self.obs_idxs:
            fig, ax = plt.subplots()
            ax.plot(self.pos_plotting_pts[i]['timesteps'], self.pos_plotting_pts[i]['avg'], label="Avg", marker='+')
            ax.plot(self.pos_plotting_pts[i]['timesteps'], self.pos_plotting_pts[i]['max'], label="Max", marker='+')
            ax.plot(self.pos_plotting_pts[i]['timesteps'], self.pos_plotting_pts[i]['min'], label="Min", marker='+')

            if title:
                ax.set_title(title)
            else:
                ax.set_title(f"Tracking Index: {i} of Observation Space")
            ax.legend()
            ax.set_xlabel("Number of Timesteps")
            ax.set_ylabel("Position Value")
            ax.grid()
            fig.text(0, -0.16, logging_note, ha='left', va='top', transform=ax.transAxes, fontsize=8)
            # Auto-layout to avoid clipping
            fig.tight_layout()

            # Then adjust bottom to fit the footer
            fig.subplots_adjust(bottom=0.16)

            self.pos_fig_list.append((i, fig))

    def plot_outcome(self, title: str = None):

        self.outcome_fig_list = []
        logging_note = f'{self.today} | Experiment No. {self.expt_num} | Learning Algo: {self.learning_algo} | Environment ID: {self.env_id} |'

        key = self.obs_idxs[0]
        x = self.pos_plotting_pts[key]['timesteps']
        bar_width = x[1] - (x[0] * 1.1)
        x = np.array(x)
        truncation_arr = np.array(self.trun_plotting_cnt)
        termination_arr = np.array(self.term_plotting_cnt)
        fig, ax = plt.subplots()

        ax.bar(x, height=truncation_arr, width=bar_width, align='edge', label="Truncation")
        ax.bar(x, height=termination_arr, bottom=truncation_arr, width=bar_width, align='edge', label="Termination")

        if title:
            ax.set_title(title)
        else:
            ax.set_title("Ratio of Truncation vs. Termination of each Rollout during Training")
        ax.legend()
        ax.set_xlabel("Number of Timesteps")
        ax.set_ylabel("Truncation Count | Termination Count")
        ax.grid(axis='y', linestyle='--', linewidth=0.5)
        fig.text(0, -0.16, logging_note, ha='left', va='top', transform=ax.transAxes, fontsize=8)
        # Auto-layout to avoid clipping
        fig.tight_layout()

        # Then adjust bottom to fit the footer
        fig.subplots_adjust(bottom=0.16)

        self.outcome_fig_list.append((0, fig))

    def save_plot(self, plot_type: str):

        try:
            if plot_type == "position":
                save_directory = os.path.join(self.env_id, "position_plots")
                fig_list = self.pos_fig_list
            elif plot_type == "reward":
                save_directory = os.path.join(self.env_id, "reward_plots")
                fig_list = self.reward_fig_list
            elif plot_type == "outcomes":
                save_directory = os.path.join(self.env_id, "outcome_plots")
                fig_list = self.outcome_fig_list
        except Exception as e:
            print(f"Error determining save directory: {e}\nplot_type must be 'position', 'reward', or 'outcomes'")
            return
        os.makedirs(save_directory, exist_ok=True)

        for i, fig in fig_list:
            extension = ".png" if platform.system() != "Windows" else ".jpg"
            if plot_type == "position":
                filename = f"{self.today}_{self.learning_algo}_expt{self.expt_num}_obs[{str(i)}]{extension}"
            else:
                filename = f"{self.today}_{self.learning_algo}_expt{self.expt_num}{extension}"

            full_path = os.path.join(save_directory, filename)
            try:
                fig.savefig(full_path)
                plt.close(fig)
                print(f"Plot saved to: {full_path}")

            except Exception as e:
                print(f"Error saving plot: {e}")
