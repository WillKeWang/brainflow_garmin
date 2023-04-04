import argparse
import logging

import pyqtgraph as pg
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from pyqtgraph.Qt import QtGui, QtCore

# added libraries

import numpy as np
import pandas as pd
import emd
import time
import datetime
from scipy import signal
from scipy.integrate import simps

def calculate_band_power(x, low, high, sf=200):
    win = 4 * sf
    freqs, psd = signal.welch(x, sf, nperseg=win)
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    freq_res = freqs[1] - freqs[0]
    band_power = simps(psd[idx_band], dx=freq_res)
    total_power = simps(psd, dx=freq_res)
    band_rel_power = band_power / total_power
    return band_power, band_rel_power

def band_power_features(segment_df):
    # minute_df must contain filtered signals from all channels
    features_dict = {}
    frequency_bands = {"delta": {'low': 1, 'high': 4},
                       "theta": {'low': 4, 'high': 8},
                       "alpha": {'low': 8, 'high': 12},
                       "beta": {'low': 12, 'high': 25},
                       }
    # features for channel_1
    delta_power_1, delta_rel_power_1 = calculate_band_power(
        x=segment_df['filtered_Channel_1'],
        low=frequency_bands['delta']['low'],
        high=frequency_bands['delta']['high']
    )
    theta_power_1, theta_rel_power_1 = calculate_band_power(
        x=segment_df['filtered_Channel_1'],
        low=frequency_bands['theta']['low'],
        high=frequency_bands['theta']['high']
    )
    alpha_power_1, alpha_rel_power_1 = calculate_band_power(
        x=segment_df['filtered_Channel_1'],
        low=frequency_bands['alpha']['low'],
        high=frequency_bands['alpha']['high']
    )
    beta_power_1, beta_rel_power_1 = calculate_band_power(
        x=segment_df['filtered_Channel_1'],
        low=frequency_bands['beta']['low'],
        high=frequency_bands['beta']['high']
    )

    features_dict.update({'Channel_1_delta_power':      delta_power_1,
                          'Channel_1_delta_rel_power':  delta_rel_power_1,
                          'Channel_1_theta_power':      theta_power_1,
                          'Channel_1_theta_rel_power':  theta_rel_power_1,
                          'Channel_1_alpha_power':      alpha_power_1,
                          'Channel_1_alpha_rel_power':  alpha_rel_power_1,
                          'Channel_1_beta_power':       beta_power_1,
                          'Channel_1_beta_rel_power':   beta_rel_power_1,
                          })

    # features for channel_2
    delta_power_2, delta_rel_power_2 = calculate_band_power(
        x=segment_df['filtered_Channel_2'],
        low=frequency_bands['delta']['low'],
        high=frequency_bands['delta']['high']
    )
    theta_power_2, theta_rel_power_2 = calculate_band_power(
        x=segment_df['filtered_Channel_2'],
        low=frequency_bands['theta']['low'],
        high=frequency_bands['theta']['high']
    )
    alpha_power_2, alpha_rel_power_2 = calculate_band_power(
        x=segment_df['filtered_Channel_2'],
        low=frequency_bands['alpha']['low'],
        high=frequency_bands['alpha']['high']
    )
    beta_power_2, beta_rel_power_2 = calculate_band_power(
        x=segment_df['filtered_Channel_2'],
        low=frequency_bands['beta']['low'],
        high=frequency_bands['beta']['high']
    )

    features_dict.update({'Channel_2_delta_power':      delta_power_2,
                          'Channel_2_delta_rel_power':  delta_rel_power_2,
                          'Channel_2_theta_power':      theta_power_2,
                          'Channel_2_theta_rel_power':  theta_rel_power_2,
                          'Channel_2_alpha_power':      alpha_power_2,
                          'Channel_2_alpha_rel_power':  alpha_rel_power_2,
                          'Channel_2_beta_power':       beta_power_2,
                          'Channel_2_beta_rel_power':   beta_rel_power_2,
                          })

    # features for channel_3
    delta_power_3, delta_rel_power_3 = calculate_band_power(
        x=segment_df['filtered_Channel_3'],
        low=frequency_bands['delta']['low'],
        high=frequency_bands['delta']['high']
    )
    theta_power_3, theta_rel_power_3 = calculate_band_power(
        x=segment_df['filtered_Channel_3'],
        low=frequency_bands['theta']['low'],
        high=frequency_bands['theta']['high']
    )
    alpha_power_3, alpha_rel_power_3 = calculate_band_power(
        x=segment_df['filtered_Channel_3'],
        low=frequency_bands['alpha']['low'],
        high=frequency_bands['alpha']['high']
    )
    beta_power_3, beta_rel_power_3 = calculate_band_power(
        x=segment_df['filtered_Channel_3'],
        low=frequency_bands['beta']['low'],
        high=frequency_bands['beta']['high']
    )

    features_dict.update({'Channel_3_delta_power':      delta_power_3,
                          'Channel_3_delta_rel_power':  delta_rel_power_3,
                          'Channel_3_theta_power':      theta_power_3,
                          'Channel_3_theta_rel_power':  theta_rel_power_3,
                          'Channel_3_alpha_power':      alpha_power_3,
                          'Channel_3_alpha_rel_power':  alpha_rel_power_3,
                          'Channel_3_beta_power':       beta_power_3,
                          'Channel_3_beta_rel_power':   beta_rel_power_3,
                          })

    # features for channel_4
    delta_power_4, delta_rel_power_4 = calculate_band_power(
        x=segment_df['filtered_Channel_4'],
        low=frequency_bands['delta']['low'],
        high=frequency_bands['delta']['high']
    )
    theta_power_4, theta_rel_power_4 = calculate_band_power(
        x=segment_df['filtered_Channel_4'],
        low=frequency_bands['theta']['low'],
        high=frequency_bands['theta']['high']
    )
    alpha_power_4, alpha_rel_power_4 = calculate_band_power(
        x=segment_df['filtered_Channel_4'],
        low=frequency_bands['alpha']['low'],
        high=frequency_bands['alpha']['high']
    )
    beta_power_4, beta_rel_power_4 = calculate_band_power(
        x=segment_df['filtered_Channel_4'],
        low=frequency_bands['beta']['low'],
        high=frequency_bands['beta']['high']
    )

    features_dict.update({'Channel_4_delta_power':      delta_power_4,
                          'Channel_4_delta_rel_power':  delta_rel_power_4,
                          'Channel_4_theta_power':      theta_power_4,
                          'Channel_4_theta_rel_power':  theta_rel_power_4,
                          'Channel_4_alpha_power':      alpha_power_4,
                          'Channel_4_alpha_rel_power':  alpha_rel_power_4,
                          'Channel_4_beta_power':       beta_power_4,
                          'Channel_4_beta_rel_power':   beta_rel_power_4,
                          })
    return features_dict


class Relative_Power_Features:
    def __init__(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate

    def calculate(self):
        frequency_bands = {"delta": {'low': 1, 'high': 4},
                           "theta": {'low': 4, 'high': 8},
                           "alpha": {'low': 8, 'high': 12},
                           "beta": {'low': 12, 'high': 25},
                           }
        data = self.board_shim.get_current_board_data(self.num_points)
        for count, channel in enumerate(self.exg_channels):
            # plot timeseries
            DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(data[channel], self.sampling_rate, 3.0, 45.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 48.0, 52.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 58.0, 62.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            
            # features for channel_1
            delta_power, delta_rel_power = calculate_band_power(
                x=data[channel],
                low=frequency_bands['delta']['low'],
                high=frequency_bands['delta']['high']
            )
            theta_power, theta_rel_power = calculate_band_power(
                x=data[channel],
                low=frequency_bands['theta']['low'],
                high=frequency_bands['theta']['high']
            )
            alpha_power, alpha_rel_power = calculate_band_power(
                x=data[channel],
                low=frequency_bands['alpha']['low'],
                high=frequency_bands['alpha']['high']
            )
            beta_power, beta_rel_power = calculate_band_power(
                x=data[channel],
                low=frequency_bands['beta']['low'],
                high=frequency_bands['beta']['high']
            )
            features_dict = {'delta_rel_power': delta_rel_power,
                             'theta_rel_power': theta_rel_power,
                             'alpha_rel_power': alpha_rel_power,
                             'beta_rel_power': beta_rel_power,
                             'alpha_beta_ratio': alpha_power/beta_power
                             }
            print(features_dict)


class Graph:
    def __init__(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate

        self.app = QtGui.QApplication([])
        self.win = pg.GraphicsWindow(title='BrainFlow Plot', size=(800, 600))

        self._init_timeseries()

        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(self.update_speed_ms)
        QtGui.QApplication.instance().exec_()

    def _init_timeseries(self):
        self.plots = list()
        self.curves = list()
        for i in range(len(self.exg_channels)):
            p = self.win.addPlot(row=i, col=0)
            p.showAxis('left', False)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            if i == 0:
                p.setTitle('TimeSeries Plot')
            self.plots.append(p)
            curve = p.plot()
            self.curves.append(curve)

    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points)
        for count, channel in enumerate(self.exg_channels):
            # plot timeseries
            DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(data[channel], self.sampling_rate, 3.0, 45.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 48.0, 52.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 58.0, 62.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            self.curves[count].setData(data[channel].tolist())

        self.app.processEvents()


def main():
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
    parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                        default=0)
    parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=0)
    parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
                        default=0)
    parser.add_argument('--ip-address', type=str, help='ip address', required=False, default='')
    parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='')
    parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
    parser.add_argument('--other-info', type=str, help='other info', required=False, default='')
    parser.add_argument('--streamer-params', type=str, help='streamer params', required=False, default='')
    parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
    parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                        required=False, default=BoardIds.SYNTHETIC_BOARD)
    parser.add_argument('--file', type=str, help='file', required=False, default='')
    parser.add_argument('--master-board', type=int, help='master board id for streaming and playback boards',
                        required=False, default=BoardIds.NO_BOARD)
    args = parser.parse_args()

    params = BrainFlowInputParams()
    params.ip_port = args.ip_port
    params.serial_port = args.serial_port
    params.mac_address = args.mac_address
    params.other_info = args.other_info
    params.serial_number = args.serial_number
    params.ip_address = args.ip_address
    params.ip_protocol = args.ip_protocol
    params.timeout = args.timeout
    params.file = args.file
    params.master_board = args.master_board

    board_shim = BoardShim(args.board_id, params)
    try:
        board_shim.prepare_session()
        board_shim.start_stream(450000, args.streamer_params)
        # Graph(board_shim)
        feature_engine = Relative_Power_Features(board_shim)
        feature_engine.calculate()
    except BaseException:
        logging.warning('Exception', exc_info=True)
    finally:
        logging.info('End')
        if board_shim.is_prepared():
            logging.info('Releasing session')
            board_shim.release_session()


if __name__ == '__main__':
    main()
