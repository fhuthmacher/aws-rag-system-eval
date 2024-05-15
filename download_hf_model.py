import os
import tempfile
from pathlib import Path

import s3fs # pip install s3fs
import smart_open
from huggingface_hub import snapshot_download


def huggingface_to_s3mirror(repo_id, s3_path):
    s3 = s3fs.S3FileSystem(anon=False)

    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_dir = Path(tmpdirname)
        snapshot_download(repo_id=repo_id, cache_dir=temp_dir)
        s3.put(str(temp_dir)+'/*', s3_path, recursive=True)
       
def huggingmirror_to_local(repo_id, s3_path, local_path=os.path.abspath(os.path.dirname('.'))):
    s3 = s3fs.S3FileSystem(anon=False)
    # Downloads from mirror to a local directory.
    s3_url = s3_path + "models--" + repo_id.replace("/", "--")
    s3.get(s3_url, lpath=local_path, recursive=True)

# s3://felixh-sagemaker/bedrock/custom_models/nllb-200-distilled-600M/
huggingface_to_s3mirror("facebook/nllb-200-distilled-600M", "s3://felixh-sagemaker/bedrock/custom_models/nllb-200-distilled-600M/")
# huggingmirror_to_local("facebook/nllb-200-distilled-600M", "s3://mybucket")