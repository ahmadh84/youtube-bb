
########################################################################
# YouTube BoundingBox Downloader
########################################################################
#
# This script downloads all videos within the YouTube BoundingBoxes
# dataset and cuts them to the defined clip size. It is accompanied by
# a second script which decodes the videos into single frames.
#
# Author: Mark Buckler
#
########################################################################
#
# The data is placed into the following directory structure:
#
# dl_dir/videos/d_set/class_id/clip_name.mp4
#
########################################################################

# from __future__ import unicode_literals
# import imageio
# from ffmpy import FFmpeg
from subprocess import check_call
from concurrent import futures
# from random import shuffle
import subprocess
# import youtube_dl
# import socket
import os
import argparse
import collections
# import io
import sys
import csv
import tempfile
import re


yt_dl_path = '/usr/local/anaconda3/bin/youtube-dl'

# The data sets to be downloaded
d_sets = ['yt_bb_classification_train',
          'yt_bb_classification_validation',
          'yt_bb_detection_train',
          'yt_bb_detection_validation']

# Host location of segment lists
web_host = 'https://research.google.com/youtube-bb/'

# these options are set from argparse
# this is the minimum gap that has to be there between two consecutive 
# annotations to create a separate clip
split_t_th = 3000
# this is the starting and ending gap before the first and last annotations
# that need to be put in for a clip
gap_t_start = 1000
gap_t_end = 1000


class FrameAnnosInfo(object):
  def __init__(self, yt_id, timestamp):
    self.yt_id = yt_id
    self.timestamp = int(timestamp)
    self.class_infos = {}
    self.obj_infos = {}

  def add_class_info(self, class_id, class_name, present):
    class_id = int(class_id)
    if class_id in self.class_infos:
      raise Exception("Class '%d' has already been added to %s (%d)" % (class_id, self.yt_id, self.timestamp))
    self.class_infos[class_id] = {'class_name':class_name, 'present':present}

  def add_obj_info(self, class_id, class_name, present, obj_id, xmin, xmax, ymin, ymax):
    obj_id = int(obj_id)
    class_id = int(class_id)
    obj_key = (obj_id, class_id)
    if obj_key in self.obj_infos:
      raise Exception("Object '%s' has already been added to %s (%d)" % (str(obj_key), self.yt_id, self.timestamp))
    self.obj_infos[obj_key] = {'class_name':class_name, 'present':present, \
                               'coords':(float(xmin), float(xmax), float(ymin), float(ymax))}

  def get_all_obj_keys(self):
    return self.obj_infos.keys()

  def gen_class_info_strs(self):
    strs = []
    for class_id, class_info in self.class_infos.items():
      clss_str = "%d,%s,%d" % (class_id, class_info['class_name'], class_info['present'])
      strs.append(clss_str)
    return strs

  def gen_obj_info_strs(self):
    strs = []
    for obj_key, obj_info in self.obj_infos.items():
      obj_str = "%d,%s,%d,%d,%f,%f,%f,%f" % (obj_key[1], obj_info['class_name'], obj_key[0], \
                                             obj_info['present'], obj_info['coords'][0], \
                                             obj_info['coords'][1], obj_info['coords'][2], \
                                             obj_info['coords'][3])
      strs.append(obj_str)
    return strs
  def verify(self):
    # verify all the coordinates
    for obj_key, obj_info in self.obj_infos.items():
      coords = obj_info['coords']
      checks = [c == -1 or (c >= 0 and c <= 1) for c in coords]
      if not (coords[0] <= coords[1] and coords[2] <= coords[3] and sum(checks)==4):
        raise Exception("Object '%s' %s (%d) has invalid coords (%s)" % (str(obj_key), self.yt_id, self.timestamp, str(coords)))

  def get_ts(self):
    return self.timestamp


# Video clip class
class Video(object):
  def __init__(self, yt_id, dl_path, train_or_val):
    self.yt_id    = yt_id
    self.dwnld_path = dl_path

    self.train_or_val = train_or_val
    self.start_ts = float('Inf')
    self.stop_ts  = -float('Inf')
    self.fps = -1
    self.num_frames = -1
    self.width = -1
    self.height = -1
    self.frame_annos = {}

  def get_download_dir(self):
    return self.dwnld_path

  def get_yt_id(self):
    return self.yt_id

  def vid_filename(self):
    return self.yt_id + '_temp.mp4'

  def get_yt_link(self):
    return 'youtu.be/' + self.yt_id

  def create_frame_anno(self, timetamp):
    timetamp = int(timetamp)
    if timetamp not in self.frame_annos:
      self.frame_annos[timetamp] = FrameAnnosInfo(self.yt_id, timetamp)
    self.start_ts = min(self.start_ts, timetamp)
    self.stop_ts  = max(self.stop_ts, timetamp)
    return timetamp

  def add_class_info(self, timetamp, *args):
    ts = self.create_frame_anno(timetamp)
    self.frame_annos[ts].add_class_info(*args)

  def add_obj_info(self, timetamp, *args):
    ts = self.create_frame_anno(timetamp)
    self.frame_annos[ts].add_obj_info(*args)

  def get_all_ts(self):
    all_ts = self.frame_annos.keys()
    all_ts = sorted(all_ts)
    return all_ts

  def gen_all_ts_groups(self):
    # compute the gap between all timestamps
    all_ts = self.get_all_ts()
    ts_diffs = [ts2-ts1 for ts2, ts1 in zip(all_ts[1:], all_ts[:-1])]
    # group the timestamps if the consecutive time difference <= split_t_th
    ts_groups = []
    curr_ts_grp = []
    for i, ts_diff in enumerate(ts_diffs):
      curr_ts_grp.append(all_ts[i])
      if ts_diff > split_t_th:
        ts_groups.append(curr_ts_grp)
        curr_ts_grp = []
    curr_ts_grp.append(all_ts[-1])
    ts_groups.append(curr_ts_grp)
    return ts_groups

  def gen_all_clips(self):
    ts_grps = self.gen_all_ts_groups()
    clips = []
    # iterate over all timestamp groups, and create separate clips for each
    for ts_grp in ts_grps:
      start_ts = max(min(ts_grp) - gap_t_start, 0)
      end_ts = max(ts_grp) + gap_t_end
      clip = {'start_ts':start_ts, 'end_ts':end_ts}
      annos = {}
      for ts in ts_grp:
        annos[ts] = self.frame_annos[ts]
      clip['clip_annos'] = annos
      clips.append(clip)
    return clips

  def get_clip_filename(self, clip):
    return "%s__%.3f--%.3f.mp4" % \
            (self.yt_id, float(clip['start_ts']), float(clip['end_ts']))

  def get_num_clips(self):
    return len(self.gen_all_ts_groups())

  def get_vid_name(self):
    return self.yt_id + '+' + str(self.start_ts) + '+' + str(self.stop_ts)

  def num_anno_frames(self):
    return len(self.frame_annos)

  def num_objs(self):
    all_obj_keys = []
    for _, anno in self.frame_annos.items():
      all_obj_keys += anno.get_all_obj_keys()
    return len(set(all_obj_keys))

  def check_all_infos(self):
    for _, anno in self.frame_annos.items():
      anno.verify()

  def is_train(self):
    return self.train_or_val == "train"

  def set_vid_info(self, vid_info):
    self.fps = vid_info['fps']
    self.num_frames = vid_info['num_frames']
    self.width = vid_info['width']
    self.height = vid_info['height']

def get_vid_info(vid_fp):
  """Gets information about a video using ffprobe:
  1. Get location of iframes
  2. get number of frames
  3. height/width of frames
  """
  vid_info = {}
  err_fd = tempfile.TemporaryFile()
  check_call(["ffmpeg", "-i", vid_fp, \
                        "-vf", r'select=eq(pict_type\,PICT_TYPE_I)', \
                        "-vsync", "2", "-f", "null", "NUL", \
                        "-loglevel", "debug"], stderr=err_fd)
  # check_call(["ffprobe", "-select_streams", "v:0", \
  #                   "-show_frames", "-show_entries", \
  #                   "frame=pict_type", "-of", "csv", vid_fp],
  #          stdout=out_fd, stderr=err_fd)

  err_fd.seek(0)
  err = err_fd.read()
  err_fd.close()

  matches = re.findall(r"pict_type:(\S+)", err)
  vid_info['num_frames'] = len(matches)
  vid_info['iframes'] = [i for i, f in enumerate(matches) if f == 'I']

  for line in err.splitlines():
    vid_line = re.search(r"\s+Stream.*Video", line)
    if vid_line:
      # get fps
      fps = re.findall(r"(\d+(?:\.\d+)?) fps", line)
      assert len(fps) == 1, "Can't find fps for " + vid_fp
      vid_info['fps'] = float(fps[0])
      # get height/width
      hw = re.findall(r"([1-9]\d*)x([1-9]\d+)", line)
      assert len(hw) >= 1, "Can't find WxH for " + vid_fp
      vid_info['width'] = int(hw[0][0])
      vid_info['height'] = int(hw[0][1])
      break
  return vid_info


def adjust_clip(clip, vid_info, clip_info):
  start_fn = clip['start_ts'] / 1000.0
  start_fn = int(start_fn * vid_info['fps'])

  # search for this start_fn in iframes
  sel_iframe = len(vid_info["iframes"]) - 1
  for i in range(len(vid_info["iframes"]) - 1):
    if vid_info["iframes"][i] <= start_fn and start_fn < vid_info["iframes"][i+1]:
      sel_iframe = i
      break

  # find timestamp for iframe
  iframe_start_sec = float(vid_info["iframes"][sel_iframe]) / vid_info['fps']

  clip_info['start_keyframe_num'] = vid_info["iframes"][sel_iframe]
  clip_info['start_msec'] = iframe_start_sec * 1000
  clip_info['end_msec'] = clip['end_ts']


# Download and cut a clip to size
def dl_and_cut(vid):
  err_msg = ""
  vid_info = None
  try:
    d_dir = vid.get_download_dir()
    vid_fp = os.path.join(d_dir, vid.vid_filename())
    clips = vid.gen_all_clips()
    clip_infos = []

    # Use youtube_dl to download the video
    FNULL = open(os.devnull, 'w')
    dwnld_cmd = [yt_dl_path, \
                 #'--no-progress', \
                 '-f', 'best[ext=mp4]', \
                 '-o', vid_fp, \
                  vid.get_yt_link()]
    # print(dwnld_cmd)
    check_call(dwnld_cmd, stdout=FNULL, stderr=subprocess.STDOUT)

    if os.path.exists(vid_fp):
      # get info about the video
      vid_info = get_vid_info(vid_fp)
      # vid_info2 = get_vid_info_old(vid_fp)
      # if vid_info != vid_info2:
      #   raise Exception("video infos are different")
      # break the video into clips
      for clip in clips:
        clip_info = {}
        # adjust the clip timing according to key-frames
        adjust_clip(clip, vid_info, clip_info)

        # get the information necessary for creating this clip
        clip_fn = vid.get_clip_filename(clip)
        clip_fp = os.path.join(d_dir, clip_fn)

        start_sec = clip_info['start_msec'] / 1000
        clip_secs = (clip_info['end_msec'] / 1000) - start_sec

        clip_info['filename'] = clip_fn
        clip_info['est_num_frames'] = int(clip_secs * vid_info['fps'])

        # use ffmpeg to clip the video without re-encoding
        err_fd = tempfile.TemporaryFile()
        cmd = ["ffmpeg", "-ss", "%.6f"%start_sec, "-i", vid_fp,
               "-t", "%.6f"%clip_secs, "-c", "copy", clip_fp, "-y"]
        check_call(cmd, stderr=err_fd)

        err_fd.seek(0)
        err = err_fd.read()
        err_fd.close()

        # adjust estimated number of frames from the output of ffmpeg
        match = re.findall(r"frame=\s+(\d+)\s+fps", err)
        if match:
          clip_info['est_num_frames'] = \
            min(clip_info['est_num_frames'], int(match[0]))

        clip_infos.append(clip_info)

    # Remove the main video (since we have the clips)
    os.remove(vid_fp)
  except Exception, e:
    err_msg = str(e)

  return vid.get_yt_id(), err_msg, vid_info, clip_infos


def write_info_to_file(fd, vid, clip_infos):
  """This function writes information about all annotation information related
  to a video to the file descriptor fd. This includes all the information
  about clips produced from this video, and the annotations the reside in that
  clip. This crucially writes time offsets and fps for each clip/video.
  """
  # write video level information
  fd.write("YTID:%s %s %dx%d %d %.2f \n" % (vid.get_yt_id(), \
                                       vid.train_or_val, \
                                       vid.height, vid.width, \
                                       vid.num_frames, \
                                       vid.fps))
  # write information about each clip
  clips = vid.gen_all_clips()
  assert len(clip_infos) == len(clips), "Inconsistent number of clip " + vid.get_yt_id()
  for i, clip in enumerate(clips):
    clip_info = clip_infos[i]
    # first write information about this clip
    fd.write("\t%s %.2f %.2f %d %d\n" % (clip_info['filename'], \
                                         clip_info['start_msec'], \
                                         clip_info['end_msec'], \
                                         clip_info['est_num_frames'], \
                                         clip_info['start_keyframe_num']))
    assert clip_info['start_msec'] <= clip['start_ts'] and \
           clip_info['end_msec'] >= clip['end_ts'], \
           "Mismatched clip timings " + vid.get_yt_id()
    clip_ts = clip['clip_annos'].keys()
    clip_ts = sorted(clip_ts)
    for ts in clip_ts:
      assert clip_info['start_msec'] <= ts and \
             clip_info['end_msec'] >= ts, \
             "Mismatched frame timings " + vid.get_yt_id()

      class_strs = clip['clip_annos'][ts].gen_class_info_strs()
      for class_str in class_strs:
        fd.write("\t\tCLASS:%d,%s\n" % (ts, class_str))

      obj_strs = clip['clip_annos'][ts].gen_obj_info_strs()
      for obj_str in obj_strs:
        fd.write("\t\tOBJ:%d,%s\n" % (ts, obj_str))

  fd.flush()


# Parse the annotation csv file and schedule downloads and cuts
def parse_and_dwnld_vids(args):
  """Download the entire youtube-bb data set into `dl_dir`.
  """
  dl_dir = args.downloaddir
  num_threads = args.numthreads

  # Make the download directory if it doesn't already exist
  if not os.path.exists(dl_dir):
    os.makedirs(dl_dir)

  # Video download directory
  vids_dir = os.path.join(dl_dir, "videos")
  if not os.path.exists(vids_dir):
    os.makedirs(vids_dir)

  videos = {}

  # For each of the four datasets
  for d_set in d_sets:
    csv_path = os.path.join(dl_dir, d_set+'.csv')

    if not os.path.isfile(csv_path):
      # Download & extract the annotation list
      print (d_set+': Downloading annotations...')
      check_call(['wget', web_host+d_set+'.csv.gz'])
      print (d_set+': Unzipping annotations...')
      check_call(['gzip', '-d', '-f', d_set+'.csv.gz'])

      os.rename(d_set+'.csv', csv_path)

    if ('classification' in d_set):
      class_or_det = 'class'
    elif ('detection' in d_set):
      class_or_det = 'det'
    else:
      raise Exception("Unknown csv type '%s'" % d_set)

    if ('train' in d_set):
      train_or_val = 'train'
      is_train = True
    elif ('validation' in d_set):
      train_or_val = 'val'
      is_train = False
    else:
      raise Exception("Unknown train/val type '%s'" % d_set)

    print (csv_path+': Parsing file ...')
    # Parse csv data
    with open(csv_path, 'rt') as f:
      reader      = csv.reader(f)
      annotations = list(reader)

    # print(annotations[0])
    # print(annotations[-1])

    print (csv_path+': Inserting in videos dict ...')

    # Parse all the annotation lines in this file - adding class or detection
    # information to Video objects in videos
    for idx, anno in enumerate(annotations):
      yt_id    = anno[0]
      timetamp = anno[1]

      # if video doesn't exist create it
      if yt_id not in videos:
        videos[yt_id] = Video(yt_id, vids_dir, train_or_val)

      if videos[yt_id].is_train() != is_train:
        raise Exception("%s has already been marked %s for training" % (yt_id, str(videos[yt_id].is_train())))

      if (class_or_det == 'class'):
        class_id = anno[2]
        class_name = anno[3]
        obj_presence = anno[4]
        obj_presence = obj_presence == "present"
        videos[yt_id].add_class_info(timetamp, class_id, class_name, obj_presence)
      elif (class_or_det == 'det'):
        class_id = anno[2]
        class_name = anno[3]
        obj_id = anno[4]
        obj_presence = anno[5]
        obj_presence = obj_presence == "present"
        coords = anno[6:]
        videos[yt_id].add_obj_info(timetamp, class_id, class_name, obj_presence, obj_id, *coords)

  print("Number of videos: %d" % len(videos))

  # delete some videos if prctdwnld given
  if args.prctdwnld < 100:
    all_vid_keys = videos.keys()
    num_del = int(len(all_vid_keys) * (100-args.prctdwnld)/100.0)
    # delete the last [num_del] videos
    vids_del = all_vid_keys[-num_del:]
    for vid in vids_del:
      del videos[vid]
    print("%d videos remain after removing %.2f%% videos" % (len(videos), 100-args.prctdwnld))

  all_num_annos = [vid.num_anno_frames() for _, vid in videos.items()]
  avg_annos = sum(all_num_annos) / float(len(videos))
  print("Avg. number of annotated frames: %.2f [%d, %d]" % (avg_annos, min(all_num_annos), max(all_num_annos)))

  all_num_objs = [vid.num_objs() for _, vid in videos.items()]
  avg_objs = sum(all_num_objs) / float(len(videos))
  print("Avg. number of tracked objects: %.2f [%d, %d]" % (avg_objs, min(all_num_objs), max(all_num_objs)))

  all_num_clips =[vid.get_num_clips() for _, vid in videos.items()]
  avg_numclips = sum(all_num_clips) / float(len(videos))
  print("Avg. number of clips: %.2f [%d, %d]" % (avg_numclips, min(all_num_clips), max(all_num_clips)))

  print('Verifying all annotations')
  for yt_id, vid in videos.items():
    vid.check_all_infos()

  fd = open("dwnld_youtubebb", "w+")
  fd_err = open("dwnld_youtubebb_err", "w+")

  # Download and cut in parallel threads giving
  with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
    fs = [executor.submit(dl_and_cut, vid) for yt_id, vid in videos.items()]
    for i, f in enumerate(futures.as_completed(fs)):
      ret_yt_id, err_msg, vid_info, clip_infos = f.result()
      if err_msg:
        fd_err.write("YTID:%s failed: %s\n" % (ret_yt_id, str(err_msg)))
        fd_err.flush()
      else:
        videos[ret_yt_id].set_vid_info(vid_info)
        write_info_to_file(fd, videos[ret_yt_id], clip_infos)

      # Write progress to stderr so far
      sys.stderr.write( \
        "Downloaded video: {} / {} \r".format(i, len(videos)))

  print('All videos (%d) downloaded' % len(videos))

  fd.close()
  fd_err.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=
"------------------  YouTube BB Donwloader ------------------"
"------------------------------------------------------------"
"This script helps downloading videos for YouTube BB: "
"https://research.google.com/youtube-bb/"
"This script has been adapted from github.com/mbuckler/youtube-bb")

  parser.add_argument("-d", "--downloaddir", type=str, metavar="DOWNLOADDIR", required=True,
                       help="Directory where the dataset is downloaded")
  parser.add_argument("-t", "--numthreads", type=int, metavar="NUMTHREADS",
                       default=4,
                       help="Number of threads to use for parallel downloading")
  parser.add_argument("-p", "--prctdwnld", type=float, metavar="PERCENTAGEDWNLD",
                       default=100,
                       help="Percentage of the dataset to download")
  parser.add_argument("-s", "--split_time_th", type=int, default=3000,
                       help="this is the minimum gap (in ms) that has to be there between "
                            "two consecutive annotations to create a separate clip.")
  parser.add_argument("-g", "--start_gap", type=int, default=1000,
                       help="this is the starting gap (in ms) that needs to be put in "
                            "before the first annotation in a clip.")
  parser.add_argument("-e", "--end_gap", type=int, default=1000,
                       help="this is the ending gap (in ms) that needs to be put in "
                            "after the last annotation in a clip.")

  args = parser.parse_args()

  split_t_th = args.split_time_th
  gap_t_start = args.start_gap
  gap_t_end = args.end_gap

  parse_and_dwnld_vids(args)
