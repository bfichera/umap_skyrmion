from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lmfit


class PowerLawPlusResidualModel(lmfit.Model):

    def __init__(self, *args, **kwargs):

        def func(x, amplitude, exponent, residual):
            return amplitude * x**exponent + residual

        super().__init__(func, *args, **kwargs)

    def guess(self, data, x, **kws):
        params = self.make_params()
        params['residual'].set(value=min(data))
        params['amplitude'].set(value=(data[-1] - data[0]) / (x[-1] - x[0]))
        params['exponent'].set(value=1.0, min=0.01)
        return params


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=lambda s: Path(s))
parser.add_argument('--output', type=lambda s: Path(s))
_cfg = parser.parse_args()
output_dir = _cfg.output
input_dir = _cfg.input_dir

model = PowerLawPlusResidualModel()
csv_fnames = [
    'num_frames_vs_total_except_img_stk_size.csv',
    'size_x_vs_total_except_img_stk_size.csv',
    'num_frames_vs_total_time.csv',
    'size_x_vs_total_time.csv'
]
for csv_path in [input_dir / fname for fname in csv_fnames]:

    with open(csv_path, 'r') as fh:
        d = pd.read_csv(fh)

    xdata = d['x'].to_numpy()
    ydata = d['y'].to_numpy()
    yerr = d['yerr'].to_numpy()
    params = model.guess(ydata, xdata)
    result = model.fit(
        ydata,
        params=params,
        weights=None,
        method='leastsq',
        x=xdata
    )
    output_path = output_dir / (csv_path.stem + '.json')
    with open(output_path, 'w') as fh:
        json.dump(
            {
                'amplitude': result.params['amplitude'].value,
                'exponent': result.params['exponent'].value,
                'residual': result.params['residual'].value
            }, fh
        )
    model_x = np.linspace(xdata[0], xdata[-1], 1000)
    plt.errorbar(xdata, ydata, yerr, marker='o')
    plt.plot(model_x, model.eval(params=result.params, x=model_x))
    plt.show()
