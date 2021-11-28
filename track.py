import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm

import torch
from tracker.multitracker import JDETracker
from utils import visualization as vis
from utils.log import logger
from utils.timer import Timer
from utils.evaluation import Evaluator
from utils.parse_config import parse_model_cfg
import utils.datasets as datasets
from utils.utils import *


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30):
    '''
       Processes the video sequence given and provides the output of tracking result (write the results in video file)

       It uses JDE model for getting information about the online targets present.

       Parameters
       ----------
       opt : Namespace
             Contains information passed as commandline arguments.

       dataloader : LoadVideo
                    Instance of LoadVideo class used for fetching the image sequence and associated data.

       data_type : String
                   Type of dataset corresponding(similar) to the given video.

       result_filename : String
                         The name(path) of the file for storing results.

       save_dir : String
                  Path to the folder for storing the frames containing bounding box information (Result frames).

       show_image : bool
                    Option for shhowing individial frames during run-time.

       frame_rate : int
                    Frame-rate of the given video.

       Returns
       -------
       (Returns are not significant here)
       frame_id : int
                  Sequence number of the last sequence
       '''

    if save_dir:
        mkdir_if_missing(save_dir)
    # 创建一个跟踪器
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    for path, img, img0 in dataloader:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1./max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        blob = torch.from_numpy(img).cuda().unsqueeze(0)
        # 返回的是跟踪到的轨迹
        online_targets = tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []
        # 遍历每一个轨迹
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6     # 如果box的width/height>1.6的话，则为false，不认为是一个行人bbox
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
        timer.toc()
        # save results
        # 保存结果，这里只是使用了一个列表暂存，还没有写入文件
        # 格式是：帧-[所有的包围框列表]-[所有的轨迹的ID列表]
        results.append((frame_id + 1, online_tlwhs, online_ids))
        # 绘制结果，所有的包围框,包围框的ID以及相应的每一帧的fps，直接一个函数搞定
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
    # 等序列中的所有图片都遍历完成之后，保存列表results至结果文件中
    # save results
    write_results(result_filename, results, data_type)
    '''该函数返回的是所有帧数，跟踪每一帧的平均时间，没有意义的一项'''
    return frame_id, timer.average_time, timer.calls

'''
    这个文件（该方法）的工作是对序列进行跟踪，输入的是一系列的图片（连续的帧），而不是一个视频，而demo.py输入的是一个视频
    并且还会调用evaluation.py进行指标评价
'''
def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo', 
         save_images=False, save_videos=False, show_image=True):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    mkdir_if_missing(result_root)
    data_type = 'mot'

    # Read config
    cfg_dict = parse_model_cfg(opt.cfg)
    opt.img_size = [int(cfg_dict[0]['width']), int(cfg_dict[0]['height'])]  # [1088, 608]

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    '''遍历每一个序列文件，比如MOT17-02-SDP'''
    for seq in seqs:
        '''output_dir = ....outputs/exp_name/seq(序列文件名)'''
        output_dir = os.path.join(data_root, '..','outputs', exp_name, seq) if save_images or save_videos else None
        print('#####################################################')
        logger.info('start seq: {}'.format(seq))
        print('#####################################################')
        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)
        '''result_filename = ....results/exp_name/seq(序列文件名)'''
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        '''读取frame_rate'''
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read() 
        frame_rate = int(meta_info[meta_info.find('frameRate')+10:meta_info.find('\nseqLength')])
        # 对于每一个seq进行eval_seq，遍历每一张图片
        nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        n_frame += nf
        # timer_avgs 保存的每一个序列跟踪每一帧的平均时间
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        ''' 创建Evaluator对象，进行性能评价MOTA'''
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
        if save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)

    # 所有的序列中的所有图片都遍历完成之后，进行指标计算
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    '''获取总的评价'''
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    '''打印总的评价指标并且保存成excel中
        保存路径为result下面的summary_{}.xlsx文件
    '''

    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='track.py')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3_1088x608.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='models/latest.pt', help='path to weights file')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.4, help='iou threshold for non-maximum suppression')
    parser.add_argument('--min-box-area', type=float, default=200, help='filter out tiny boxes')
    parser.add_argument('--track-buffer', type=int, default=30, help='tracking buffer')
    # 下面3个 'store_true', 如果不加的话，默认是false
    # 不触发的时候就是false，触发的时候是true
    parser.add_argument('--test-mot16', action='store_true', help='tracking buffer')
    parser.add_argument('--save-images', action='store_false', help='save tracking results (image)')
    parser.add_argument('--save-videos', action='store_false', help='save tracking results (video)')
    opt = parser.parse_args()
    print(opt, end='\n\n')
 
    if not opt.test_mot16:        # test_mot = false
        seqs_str = '''MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-13-SDP
                    '''
        data_root = '/home/kb249/cxf/JDE/testDatasets/MOT17/images/train'
        # seqs_str = '''MOT17-01-SDP
        #               MOT17-03-SDP
        #               MOT17-06-SDP
        #               MOT17-08-SDP
        #               MOT17-12-SDP
        #               MOT17-14-SDP
        #             '''
        # data_root = '/home/kb249/cxf/JDE/testDatasets/MOT17/test'
    else:
        seqs_str = '''MOT16-01
                     MOT16-03
                     MOT16-06
                     MOT16-07
                     MOT16-08
                     MOT16-12
                     MOT16-14'''
        data_root = '/home/kb249/cxf/JDE/testDatasets/MOT16/test'
    seqs = [seq.strip() for seq in seqs_str.split()]

    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name=opt.weights.split('/')[-2],
         show_image=False,
         save_images=opt.save_images, 
         save_videos=opt.save_videos)

