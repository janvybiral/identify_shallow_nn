"""
Simple experiment wrapper without any parallelization 
"""

import pandas
import pathlib
import sys
from datetime import datetime 
import itertools


def create_parameter_combinations(fixed_params,varying_params, depends_on_varying_param = None):
    varying_params_flat = _split_parameter_combinations(varying_params)
    if depends_on_varying_param:
            pars = [depends_on_varying_param({**fixed_params, **vp}) for vp in varying_params_flat]
    else:
            pars = [{**fixed_params, **vp} for vp in varying_params_flat]
    return pars

def _split_parameter_combinations(par_ranges):
    list(zip(*par_ranges.items()))
    keys, values = zip(*par_ranges.items())
    pars = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return pars

class Experiment():
    def __init__(
        self,
        run_once,
        fixed_params,
        varying_params,
        host_config,
        handle,
        depends_on_varying_param = None,
        use_pickle = False,
        return_models = False
        ):
        
        self.parse_host_config(host_config)
        #Handle for the run
        self.handle = handle
        #Experiment Setup
        self.pars = create_parameter_combinations(
            fixed_params=fixed_params,
            varying_params=varying_params,
            depends_on_varying_param = depends_on_varying_param
        )

        self.use_pickle = use_pickle
        self.run_with_parameter = run_once

    def parse_host_config(self, host_config):
        output_dir = pathlib.Path(host_config["output_dir"])
        if not output_dir.is_dir():
            print("Output path given in host_config is invalid. Exiting...")
            sys.exit()
        self.output_dir = output_dir
        #self.output_path = output_dir / host_config["file_name"]

    def __call__(self):
        results = []
        for par in self.pars:
            try:
                results.append(self.run_with_parameter(**par))
            except (RuntimeError, TypeError, NameError) as e:
                print("Error during experiment at parameter combiniation: {}".format(par))
                print("Error message: {}".format(str(e)))
        rows= [ pandas.concat([pandas.Series(par), res]) for (par, res) in zip(self.pars, results)]
        df = pandas.DataFrame(data=rows)
        if len(results) > 0: #Only save the dataframe if at leat on run worked
            timestr = datetime.now().strftime('%Y%m%d_%H%M%S')
            if not self.use_pickle:
                df.to_hdf(self.output_dir, key = "{}_{}".format(self.handle, timestr))
            elif self.use_pickle:
                df.to_pickle( self.output_dir / "{}_{}.pkl".format(self.handle, timestr))
        return df