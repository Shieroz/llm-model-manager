from pydantic import BaseModel


class ModelSetup(BaseModel):
    hf_repo: str
    quant: str
    mmproj: str = ""
    symlink_name: str
    original_name: str = ""
    parameters: str
    revision: str = "latest"
    # MTP draft head (separate-head models, same repo as the main quant). rfilename of
    # the head GGUF, or "" for grafted/no-MTP models. Cross-repo heads are not managed
    # here — users wire those manually via params + a served symlink path.
    mtp_head: str = ""


class RevisionDeleteReq(BaseModel):
    repo: str
    revision: str


class RpcModeReq(BaseModel):
    enabled: bool
