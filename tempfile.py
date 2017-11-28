import tempile
import json
import os

def write_tmp_json(cont):
    tmp = tempfile.mktemp()
    with open(tmp,"w") as f:
        f.write(json.dumps(cont))
    return tmp

def parse_tmp_json(tmp):
    with open(tmp) as f:
        ret = json.load(f)
    if isinstance(ret, dict):
        ret_ = dict()
        for k in ret:
            ret_[int(k)] = ret[k]
        return ret_
    return ret
    
backend_parse_list = dict{...}
backend_outdir_list = [..]

cmd = "cd ../../ && ./Diannei_ai_backend.sh {0:} {1:}".format(write_tmp_json(backend_parse_list),
                                                              write_tmp_json(backend_outdir_list))
                                                              
os.system(cmd)                                                             
