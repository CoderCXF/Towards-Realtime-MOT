import numpy as np
from collections import OrderedDict


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack(object):
    _count = 0
    # 轨迹id
    track_id = 0
    # 是否激活态
    is_activated = False
    # 轨迹状态
    state = TrackState.New

    history = OrderedDict()
    # 特征
    features = []
    # 当前特征
    curr_feature = None
    # 轨迹得分
    score = 0
    # 起始帧
    start_frame = 0
    # 帧id
    frame_id = 0
    # 卡尔曼滤波更新轨迹的时间
    time_since_update = 0

    # multi-camera use
    location = (np.inf, np.inf)

    # 轨迹结束帧
    @property
    def end_frame(self):
        return self.frame_id

    # 下一帧号
    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

