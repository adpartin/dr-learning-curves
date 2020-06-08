from utils.utils import verify_path

def boolify(s):
    if s=='True':
        return True
    if s=='False':
        return False
    raise ValueError("huh?")

def autoconvert(s):
    for fn in (boolify, int, float):
        try:
            return fn(s)
        except ValueError:
            pass
    return s

def read_hp_prms(hp_file):
    dct = dict()
    hp_file = verify_path( hp_file )
    with open(hp_file, 'r') as file:
        for line in file:
            aa = line.strip().split(':')
            # print(line)
            # key = aa[0].strip()
            # val = aa[1].strip()
            # val = autoconvert( val )
            # ml_init_kwargs[key] = val
            dct[aa[0].strip()] = autoconvert( aa[1].strip() )
    return dct


