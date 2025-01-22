# pipeline.py
# Eden McEwen
# this file contains the scripts to run to reduce data in a Spark obj
from sparkles.spark import *
from sparkles.file_reader import *

#run the pool and save the data
def pipe_main(data_dir, lab_data, dir_calib=glob_dir_calib, dark=glob_dark, mask=glob_mask, ref=glob_ref):
    """
    This file takes in sky and lab directories
        these need to end in camwfs/
    It will reduce them in PARALLEL, and save a npy file of dot products 
    """
    sp_data = Spark(data_dir, lab_data, dir_calib, dark, mask, ref)
    sp_lab = Spark(lab_data, lab_data, dir_calib, dark, mask, ref)
    f_list_data = file_lister(sp_data.dir_data)

    data_splits_list = []

    for n in range(len(f_list_data)//int(sp_data.data_HZ*60)):
        ni = int(n*sp_data.data_HZ*60)
        print(f"START: Block {n} starting with file {ni}!")
        try:
            data_split_s = np.array(sp_data.dot_list_pool(n_start=ni, n=1800*60, n_workers=60))
            data_splits_list.append(data_split_s)
        except Exception as e:
            print(f"We had an error at file {ni}!")
            print(e)
            continue
        print(f"=> END: Block {n}")

    # make sure to save this result!
    np.save(data_dir.replace("camwfs/", "data_splits_list.npy"), data_splits_list)
    lab_split_list = np.array(sp_lab.dot_list_pool(n=int(sp_data.data_HZ*60), n_workers=10))
    np.save(data_dir.replace("camwfs/", "lab_splits_list.npy"), lab_split_list)


def pipe_all_chunk(data_dir, lab_data, dir_calib=glob_dir_calib, dark=glob_dark, mask=glob_mask, ref=glob_ref):
    """
    This file takes in sky and lab directories
        these need to end in camwfs/
    It will reduce both in PARALLEL CHUNKS
    """
    sp_data = Spark(data_dir, lab_data, dir_calib, dark, mask, ref)
    sp_lab = Spark(lab_data, lab_data, dir_calib, dark, mask, ref)
    f_list_data = file_lister(sp_data.dir_data)

    data_splits_list = []

    for n in range(len(f_list_data)//int(sp_data.data_HZ*60)):
        ni = int(n*sp_data.data_HZ*60)
        chunk_n = int(sp_data.data_HZ/4)    
        print(f"START: Block {n} starting with file {ni}!")
        try:
            data_split_s = np.array(sp_data.dot_chunk_pool(chunk_n, 60, n_start=ni, n_workers=15))
            data_splits_list.append(data_split_s)
        except Exception as e:
            print(f"We had an error at file {ni}!")
            print(e)
            continue
        print(f"=> END: Block {n}")

    # make sure to save this result!
    np.save(data_dir.replace("camwfs/", "data_chunks_list.npy"), data_splits_list)
    lab_split_list = np.array(sp_lab.dot_list_pool(n=int(sp_data.data_HZ*60), n_workers=10))
    np.save(data_dir.replace("camwfs/", "lab_splits_c_list.npy"), lab_split_list)

def pipe_sky_chunk(data_dir, lab_data, dir_calib=glob_dir_calib, dark=glob_dark, mask=glob_mask, ref=glob_ref):
    """
    This file takes in SKY and LAB directories
        these need to end in camwfs/
    It will reduce only on sky in PARALLEL CHUNKS
    """
    sp_data = Spark(data_dir, lab_data, dir_calib, dark, mask, ref)
    f_list_data = file_lister(sp_data.dir_data)

    data_splits_list = []

    for n in range(len(f_list_data)//int(sp_data.data_HZ*60)):
        ni = int(n*sp_data.data_HZ*60)
        chunk_n = int(sp_data.data_HZ/4)
        print(f"START: Block {n} starting with file {ni}!")
        try:
            data_split_s = np.array(sp_data.dot_chunk_pool(chunk_n, 60, n_start=ni, n_workers=15))
            data_splits_list.append(data_split_s)
        except Exception as e:
            print(f"We had an error at file {ni}!")
            print(e)
            continue
        print(f"=> END: Block {n}")

    # make sure to save this result!
    np.save(data_dir.replace("camwfs/", "data_chunks_list.npy"), data_splits_list)
    return

def lab_archive(dir_lab, dir_calib=glob_dir_calib, dark=glob_dark, mask=glob_mask, ref=glob_ref, remove=True):
    """
    This file takes in a LAB dir
        this need to end in camwfs/
    It will reduce only on sky in PARALLEL CHUNKS
    If remove=TRUE, it will tidy up and delete files over 10s
    """
    if not os.path.isdir(dir_lab):
        print("There is no: ", dir_lab)
        return
    print(dir_lab)
    lab_dotseries_f = dir_lab.replace("camwfs/", "lab_splits_total_list.npy")
    if os.path.isfile(lab_dotseries_f):
        print(lab_dotseries_f, " is already created, moving on.")
        return

    sp_lab = Spark(dir_lab, dir_lab, dir_calib, dark, mask, ref)
    f_list_data = file_lister(sp_lab.dir_data)

    data_splits_list = []
    ten_sc_int = int(sp_lab.data_HZ*10)
    n_files = int(sp_lab.data_HZ*60)
    block_iter = len(f_list_data)//(n_files) + 1
    
    for n in range(block_iter): #todo; change
        ni = n*n_files
        print(f"START: Block {n} of {block_iter} starting with file {ni}!")
        try:
            data_split_s = np.array(sp_lab.dot_list_pool(n=n_files, n_workers=10))
            data_splits_list.append(data_split_s)
        except Exception as e:
            print(f"We had an error at file {ni}!")
            print(e)
            continue
        print(f"=> END: Block {n}")

    # make sure to save this result!
    lab_dotseries_f = dir_lab.replace("camwfs/", "lab_splits_total_list.npy")
    np.save(lab_dotseries_f, np.vstack(data_splits_list))

    # now save the reference
    fits_file = dir_lab.replace("camwfs/", "lab_ref.fits")
    fits.writeto(fits_file, sp_lab.labref_norm, overwrite=True)

    if remove:
        print("Removing all files over 10s: ..")
        # delete all but 1s of data:
        for f in f_list_data[ten_sc_int:]:
            os.remove(dir_lab+f)
        print(f"Done! {len( f_list_data[ten_sc_int:])} files deleted.")
    return