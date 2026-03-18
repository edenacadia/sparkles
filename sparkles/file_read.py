
import re
import pathlib
import datetime
import fixr
import lookyloo
import numpy as np
from lookyloo.core import get_matching_paths

# TODO: put this in a config file
glob_data_p = pathlib.Path('/data/rawimages/')
glob_telem_p = pathlib.Path('/srv/aoc/opt/MagAOX')

####### file pulling - only relevand for pulling observations

def verify_obs(obs_name, dt_start, n=0):
    '''
    Takes in a observation name
    Returns index n of list of obs span
        - for now, will default to first. 
        - set n to desired new place in list
    ''' 
    obs_spans_with_name = verify_obs_all(obs_name, dt_start)
    N = len(obs_spans_with_name)
    # check on list index
    if n > N:
        n = -1
    return obs_spans_with_name[n]

def verify_obs_all(obs_name, dt_start):
    '''
    Takes in a observation name
    Returns a list of obs span
        - for now, will default to first. 
        - set n to desired new place in list
    ''' 
    run_start_dt =  dt_start
    all_obs_spans, _ = lookyloo.core.get_new_observation_spans(data_roots=[glob_telem_p], existing_observation_spans=set(), start_dt=run_start_dt)
    obs_spans_with_name = [x for x in all_obs_spans if obs_name in x.title] #this can return multiple spans
    # so we mingt get maore 
    obs_spans_with_name = sorted(obs_spans_with_name, key=lambda obs: obs.begin)
    print([f'{obs.end - obs.begin} \n' for obs in obs_spans_with_name] )
    return obs_spans_with_name

def gen_file_list(obs_span, device = 'camwfs'):
    '''
    takes in a lookyloo obs span
    will then get all the matching paths for the file lsits. 
    '''
    extension = 'xrif'
    datetime_newer = obs_span.begin
    datetime_older = obs_span.end
    # checking the files associated with
    glob_data_p_dev = glob_data_p / device 
    obs_file_list = lookyloo.core.get_matching_paths(glob_data_p_dev, device, extension, datetime_newer, datetime_older)
    return sorted(obs_file_list, key=lambda obs: obs.timestamp)

def pull_n_files(file_lists, n, n_start=0, fsize=512):
    '''
    We can only pull in chunks of 512
    - take n and pull at least that many 
    - check timings
    '''
    print(f"PULLING {n} FILES")
    if n==0:
        print("This is an error!")
        return np.array([[[]]]), np.array([[]])
    # make a pull of (n//512 + 1) * 512 files
    xrif_count = (n // fsize) + 1
    n_start_xrif = n_start // fsize # how many xrif files over to shift
    n_offset = n_start % fsize # within anxrif, the offset
    print(f"FILE no {xrif_count}, n_start {n_start}, n {n}, n_offset {n_offset}")
    print(f"XRIF index {n_start_xrif}, no of files {xrif_count}, len list {len(file_lists)}")
    # make a NP array
    data_conglom = []
    timing_conglom = []
    # one by one and open
    for i in np.arange(n_start_xrif, n_start_xrif+xrif_count):
        d, ts = pull_file_xrif(file_lists[i], fsize=fsize)
        data_conglom.append(d)
        timing_conglom.append(ts)
    # stack it (only if we need to)
    if len(data_conglom) > 1:
        data_stack = np.vstack(data_conglom)[n_offset : n_offset + n]
        timing_stack = np.vstack(timing_conglom)[n_offset : n_offset + n]
    else:
        print("   => actual shape", np.array(data_conglom[0]).shape)
        print("   => index at", n_offset + n)
        data_stack = np.array(data_conglom[0])[n_offset : n_offset + n]
        timing_stack = np.array(timing_conglom[0])[n_offset : n_offset + n]
    print("file n pull", data_stack.shape)
    #todo: might need resize arrays
    return data_stack, timing_stack

def pull_file_xrif(file_obs, fsize=512):
    '''
    Take a path, return the data and the timing
    '''
    with open(file_obs.path, 'rb') as fh:
        data = fixr.xrif2numpy(fh)
        timing = fixr.xrif2numpy(fh)
    #TODO: check if this is the right approach
    n_files = int(data.size / 120 / 120)
    return np.reshape(data, (n_files,120,120)),  np.reshape(timing, (n_files,5))


def dir_to_lookyloo(obs_str, target_name="beta_pic"):
    # searching for the lookyloo format
    m = re.match(r'^(\d{4})-(\d{2})-(\d{2})_(\d{2})(\d{2})(\d{2})_(.+)$', obs_str)
    if not m:
        raise ValueError(f"Unrecognized format: {obs_str!r}")
    # break down the date into the datetime format
    Y, M, D, hh, mm, ss, rest = m.groups()
    dt = datetime.datetime(int(Y), int(M), int(D), int(hh), int(mm), int(ss), tzinfo=datetime.timezone.utc)
    
    # take the target name out of the remaining string
    if target_name:
        # if you find this string, replace it. return what's left
        if rest.find(target_name) != -1:
            rmndr = rest.replace(target_name, "")
            return dt, rmndr[1:]
    # fallback: return the first token after the datetime
    first_token = rest.split('_')[-1]
    return dt, first_token

# need to work these in: 
def xrif_list_check(n_start, n_end, l_len, n_work, fsize=512):
    ''' Converting files to xrif indexes '''
    n_start_x = n_start // fsize # how many xrif files over to shift
    n_end_x = (n_end // fsize) + 1
    # check the end number 
    if n_start_x > l_len:
        print("ERROR: n_start isn't usable")
        n_start = 0
    if n_end_x == -1:
        n_end_x = (l_len//n_work)*n_work
    elif n_end_x < n_start_x:
        print("ERROR: n_end isn't usable")
        n_end_x = n_start_x + 10 # arbitrary 10 file add
    elif n_end_x > l_len:
        print("ERROR: changing n_end to file count")
        n_end_x = (l_len//n_work)*n_work
    return n_start_x, n_end_x

def file_list_check(n_start, n_end, l_len, n_work):
    # check the end number 
    if n_start > l_len:
        print("ERROR: n_start isn't usable")
        n_start = 0
    if n_end == -1:
        n_end = (l_len//n_work)*n_work
    elif n_end < n_start:
        print("ERROR: n_end isn't usable")
        n_end = n_start + 100
    elif n_end > l_len:
        print("ERROR: changing n_end to file count")
        n_end = (l_len//n_work)*n_work