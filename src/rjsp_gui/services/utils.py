import pathlib
import tempfile
from streamlit.runtime.uploaded_file_manager import UploadedFile


def dump_to_temp(uploaded: UploadedFile, suffix: str = ".json") -> pathlib.Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.read())
    tmp.flush()
    tmp.close()
    return pathlib.Path(tmp.name)
