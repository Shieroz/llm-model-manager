from pydantic import BaseModel


class ModelSetup(BaseModel):
    hf_repo: str
    quant: str
    mmproj: str = ""
    symlink_name: str
    original_name: str = ""
    parameters: str
    revision: str = "latest"


class RevisionDeleteReq(BaseModel):
    repo: str
    revision: str


class RpcModeReq(BaseModel):
    enabled: bool
