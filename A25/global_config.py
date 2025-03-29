import os, threading
# from VISUALIZE import mcom

def get_root_dir():
    # root_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.getcwd() + '/'
    return root_dir

# def get_a_logger():
#     from VISUALIZE.mcom import mcom, logdir
#     mcv = mcom( path='%s/logger/'%logdir,
#                     digit=16,
#                     rapid_flush=True,
#                     draw_mode='Img',
#                     tag='[A25]',
#                     resume_mod=False)
#     mcv.rec_init(color='b')
#     return mcv


class GlobalConfig:
    debug = False

    root_dir = get_root_dir()
    device = 'cuda:0'
    conf_threshold = 0.3
    half = False
    # sz_wh = (1280, 578)
    sz_wh = (640, 640)

    # mcv = get_a_logger()