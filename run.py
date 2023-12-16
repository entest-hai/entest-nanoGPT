from sagemaker.pytorch import PyTorch
from sagemaker import TrainingInput
from sagemaker import Session

# get bucket
session = Session()
bucket = session.default_bucket()

# estimator 
estimator = PyTorch(
    role="",
    entry_point="train.py",
    framework_version="2.0.1",
    py_version="py310",
    instance_count=1,
    instance_type="ml.g5.12xlarge",
    hyperparameters={
      'backend': 'gloo',
      'model-type': 'custom'
    },
    distribution={
        # mpirun backend
        "pytorchddp": {"enable": True}
    },
)

# fit with s3 data
estimator.fit(
  inputs=TrainingInput(
    s3_data=f's3://{bucket}/train/input.txt',
    content_type="text/scv",
    s3_data_type="S3Prefix",
    record_wrapping=None,
    compression=None
    )
  )
