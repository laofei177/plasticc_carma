#!/opt/conda/bin/python
import numpy as np
import pandas as pd
import os, sys, glob
import dask
from notify_run import Notify
from dask.distributed import Client
from fit import *

notify = Notify()

srcDir = sys.argv[1]
outDir = sys.argv[2]

src_ls = glob.glob(os.path.join(srcDir, "*_test_lightcurves*.csv"))
bands = ["u", "g", "r", "i", "z", "y"]


if __name__ == "__main__":

    client = Client(n_workers=44)

    for i, src in enumerate(src_ls):

        # create dir for each seprate lc file
        gp_dir = os.path.join(outDir, f"lc_gp_{i}")
        if not os.path.exists(gp_dir):
            os.mkdir(gp_dir)

        # load lc into df, groupby, and get basic stat
        test_lc = pd.read_csv(src)
        lc_gps = test_lc.groupby(by="object_id")
        gps_keys = list(lc_gps.groups.keys())

        # create col names
        cols = ["object_id"]
        for band in bands:
            cols.append(f"a1_{band}")
        for band in bands:
            cols.append(f"a2_{band}")
        for band in bands:
            cols.append(f"b0_{band}")

        chunk_size = 10000

        # loop over all LC, save every 10k
        for j in range(int(len(gps_keys) / chunk_size)):

            total = []
            for k, key in enumerate(
                gps_keys[j * chunk_size : j * chunk_size + chunk_size]
            ):
                lc = lc_gps.get_group(key)

                # remove large errors
                max_err = np.percentile(lc.flux_err, 99)
                lc = lc[lc.flux_err < max_err]

                r = client.submit(carma_fit, lc, 2, 0)
                total.append(r)

            # combine result returned by dask and save
            total_rt = client.gather(total, errors="skip")
            df = pd.DataFrame(total_rt, columns=cols)
            df["object_id"] = df["object_id"].astype(int)
            df.to_csv(os.path.join(gp_dir, f"batch_{j}"), index=False)

        notify.send(f"Done with lc file: {i}")

    notify.send("All done!")

