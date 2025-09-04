from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.dtb70_path = '/media/li/D6612D9737620A9A/AVTrack/data/dtb70_path'
    settings.got10k_lmdb_path = '/media/li/D6612D9737620A9A/AVTrack/data/got10k_lmdb'
    settings.got10k_path = '/media/li/D6612D9737620A9A/AVTrack/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/media/li/D6612D9737620A9A/AVTrack/data/itb'
    settings.lasot_extension_subset_path_path = '/media/li/D6612D9737620A9A/AVTrack/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/media/li/D6612D9737620A9A/AVTrack/data/lasot_lmdb'
    settings.lasot_path = '/media/li/D6612D9737620A9A/AVTrack/data/lasot'
    settings.network_path = '/home/li/projects/AVTrack/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/media/li/D6612D9737620A9A/AVTrack/data/nfs'
    settings.otb_path = '/media/li/D6612D9737620A9A/AVTrack/data/otb'
    settings.prj_dir = '/home/li/projects/AVTrack'
    settings.result_plot_path = '/home/li/projects/AVTrack/output/test/result_plots'
    settings.results_path = '/home/li/projects/AVTrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/li/projects/AVTrack/output'
    settings.segmentation_path = '/home/li/projects/AVTrack/output/test/segmentation_results'
    settings.tc128_path = '/media/li/D6612D9737620A9A/AVTrack/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/media/li/D6612D9737620A9A/AVTrack/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/media/li/D6612D9737620A9A/AVTrack/data/trackingnet'
    settings.uav123_10fps_path = '/media/li/D6612D9737620A9A/AVTrack/data/uav123_10fps_path'
    settings.uav123_path = '/media/li/D6612D9737620A9A/AVTrack/data/uav123_path'
    settings.uav_path = '/media/li/D6612D9737620A9A/AVTrack/data/uav'
    settings.uavdt_path = '/media/li/D6612D9737620A9A/AVTrack/data/uavdt_path'
    settings.visdrone2018_path = '/media/li/D6612D9737620A9A/AVTrack/data/visdrone2018_path'
    settings.vot18_path = '/media/li/D6612D9737620A9A/AVTrack/data/vot2018'
    settings.vot22_path = '/media/li/D6612D9737620A9A/AVTrack/data/vot2022'
    settings.vot_path = '/media/li/D6612D9737620A9A/AVTrack/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

