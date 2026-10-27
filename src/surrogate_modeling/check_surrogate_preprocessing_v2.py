#!/usr/bin/env python3
'Validate completed v2 preprocessing; no regressors, tuning, or test scoring.'
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "results" / "surrogate_dataset" / "surrogate_LD18_hard_disk_extension_v2"


def main():
    marker=json.loads((DATASET_DIR/"completion_metadata.json").read_text())
    baseline_file=DATASET_DIR/"expanded_dataset.csv"
    if marker.get("completed") is not True or hashlib.sha256(baseline_file.read_bytes()).hexdigest()!=marker["output_sha256"]["expanded_dataset.csv"]:
        raise ValueError("Completed baseline changed or is incomplete.")
    baseline=pd.read_csv(baseline_file)
    expected={("main",f):200 for f in [40.,45.,50.,55.,60.]}
    expected.update({("interpolation_test",f):10 for f in [42.5,47.5,52.5,57.5]})
    expected.update({("interpolation_test_fresh",f):50 for f in [42.5,47.5,52.5,57.5]})
    if len(baseline)!=1240 or baseline["sample_id"].duplicated().any() or baseline.groupby(["split","vf_target_percent"],dropna=False).size().to_dict()!=expected:
        raise ValueError("Unexpected dataset identities/counts.")
    ids=baseline["sample_id"].astype(str).to_numpy()
    splits=baseline["split"].astype(str).to_numpy()
    descriptor_file=DATASET_DIR/"descriptor_audit"/"microstructure_descriptors.csv"
    descriptor=pd.read_csv(descriptor_file)
    metadata_file=DATASET_DIR/"two_point_statistics"/"two_point_statistics_metadata.csv"
    metadata=pd.read_csv(metadata_file)
    for name,table in [("descriptors",descriptor),("S2 metadata",metadata)]:
        if len(table)!=1240 or not np.array_equal(table["sample_id"].astype(str).to_numpy(),ids) or not np.array_equal(table["split"].astype(str).to_numpy(),splits):
            raise ValueError(f"Dataset/{name} row identities/order differ.")
        for field in ["vf_achieved","vf_target_percent"]:
            if not np.allclose(table[field].to_numpy(dtype=float),baseline[field].to_numpy(dtype=float),rtol=0,atol=1e-12):
                raise ValueError(f"Dataset/{name} {field} differs.")
    fields=["vf_achieved","nn_distance_std_over_D","local_vf_cv","centre_anisotropy",
            "C11_Pa","C12_Pa","C22_Pa","C33_Pa"]
    if not np.isfinite(descriptor[fields].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite descriptors or mechanical labels.")
    targets=["C11_Pa","C12_Pa","C22_Pa","C33_Pa"]
    if not np.allclose(descriptor[targets].to_numpy(dtype=float),baseline[targets].to_numpy(dtype=float),rtol=1e-12,atol=1e-5):
        raise ValueError("Descriptor extraction changed mechanical labels.")
    arrays={}
    paths={"raw":DATASET_DIR/"two_point_statistics"/"periodic_fibre_S2_128x128.npz",
           "normalized":DATASET_DIR/"two_point_statistics"/"periodic_fibre_normalized_S2_128x128.npz"}
    for name,path in paths.items():
        with np.load(path,allow_pickle=False) as archive:
            if not np.array_equal(archive["sample_id"].astype(str),ids) or not np.array_equal(archive["split"].astype(str),splits):
                raise ValueError(f"Dataset/{name} archive row identities/order differ.")
            if int(archive["image_resolution"][0])!=128:
                raise ValueError("Expected 128-pixel resolution.")
            for field in ["vf_achieved","vf_target_percent"]:
                if not np.allclose(archive[field],baseline[field].to_numpy(dtype=float),rtol=0,atol=1e-12):
                    raise ValueError(f"Dataset/{name} {field} differs.")
            arr=archive["S2"].copy()
            if arr.shape!=(1240,16384) or not np.isfinite(arr).all():
                raise ValueError(f"Invalid {name} S2 matrix.")
            arrays[name]=arr
    raster=metadata["raster_vf"].to_numpy(dtype=float)
    if not np.isfinite(raster).all() or np.any(raster<=0) or np.any(raster>=1):
        raise ValueError("Invalid raster fractions.")
    if not np.allclose(arrays["raw"][:,0],raster,rtol=0,atol=1e-6):
        raise ValueError("Raw S2 zero-offset mismatch.")
    expected_normalized=(arrays["raw"].astype(np.float64)-raster[:,None]**2)/(raster*(1-raster))[:,None]
    if not np.allclose(arrays["normalized"],expected_normalized,rtol=0,atol=2e-6):
        raise ValueError("Normalized S2 formula mismatch.")
    if not np.allclose(arrays["normalized"][:,0],1,rtol=0,atol=1e-5) or not np.allclose(arrays["normalized"].mean(axis=1),0,rtol=0,atol=1e-5):
        raise ValueError("Normalized S2 invariant mismatch.")
    sources={"dataset":baseline_file,"descriptors":descriptor_file,"S2_metadata":metadata_file,**paths}
    report={"completed":True,"records":1240,"development":1000,"fresh_interpolation_test":200,"existing_interpolation":40,
            "S2_shape":[1240,16384],"image_resolution":128,"models_trained":False,"fresh_test_prediction_errors_computed":False,
            "sha256":{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in sources.items()}}
    (DATASET_DIR/"preprocessing_validation_v2.json").write_text(json.dumps(report,indent=2)+"\n")
    print("PREPROCESSING VALIDATED: 1240 aligned records.")
    print("Development: 1000; fresh interpolation test: 200; existing interpolation: 40.")
    print("Raw and normalized S2 shape: (1240, 16384).")
    print("No regressors fitted; fresh interpolation-test prediction errors not computed.")


if __name__=="__main__":
    main()
