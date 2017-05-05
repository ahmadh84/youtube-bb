
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

  def add_class_info(self, class_id, present):
    class_id = int(class_id)
    if class_id in self.class_infos:
      raise Exception("Class '%d' has already been added to %s (%d)" % (class_id, self.yt_id, self.timestamp))
    self.class_infos[class_id] = {'present':present}

  def add_obj_info(self, class_id, present, obj_id, xmin, xmax, ymin, ymax):
    obj_id = int(obj_id)
    class_id = int(class_id)
    obj_key = (obj_id, class_id)
    if obj_key in self.obj_infos:
      raise Exception("Object '%s' has already been added to %s (%d)" % (str(obj_key), self.yt_id, self.timestamp))
    self.obj_infos[obj_key] = {'present':present, 'coords':(float(xmin), float(xmax), float(ymin), float(ymax))}

  def get_all_obj_keys(self):
    return self.obj_infos.keys()

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
    self.frame_annos = {}

  def get_download_dir(self):
    return self.dwnld_path

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


# Download and cut a clip to size
def dl_and_cut(vid):
  d_dir = vid.get_download_dir()
  vid_fp = os.path.join(d_dir, vid.vid_filename())
  clips = vid.gen_all_clips()

  # Use youtube_dl to download the video
  FNULL = open(os.devnull, 'w')
  dwnld_cmd = [yt_dl_path, \
               #'--no-progress', \
               '-f', 'best[ext=mp4]', \
               '-o', vid_fp, \
                vid.get_yt_link()]
  # print(dwnld_cmd)
  check_call(dwnld_cmd, stdout=FNULL, stderr=subprocess.STDOUT)

  # for clip in clips:
  #   # Verify that the video has been downloaded. Skip otherwise
  #   if os.path.exists(vid_fp):
  #     # Cut out the clip within the downloaded video and save the clip
  #     # in the correct class directory. Note that the -ss argument coming
  #     # first tells ffmpeg to start off with an I frame (no frozen start)
  #     check_call(['ffmpeg',\
  #       '-ss', str(float(clip.start)/1000),\
  #       '-i','file:'+d_set_dir+'/'+vid.yt_id+'_temp.mp4',\
  #       '-t', str((float(clip.start)+float(clip.stop))/1000),\
  #       '-c','copy',class_dir+'/'+clip.name+'.mp4'],
  #       stdout=FNULL,stderr=subprocess.STDOUT )

  # # Remove the temporary video
  # os.remove(vid_fp)


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
        obj_presence = anno[4]
        obj_presence = obj_presence == "present"
        videos[yt_id].add_class_info(timetamp, class_id, obj_presence)
      elif (class_or_det == 'det'):
        class_id = anno[2]
        obj_id = anno[4]
        obj_presence = anno[5]
        obj_presence = obj_presence == "present"
        coords = anno[6:]
        videos[yt_id].add_obj_info(timetamp, class_id, obj_presence, obj_id, *coords)

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

  # Download and cut in parallel threads giving
  with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
    fs = [executor.submit(dl_and_cut, vid) for yt_id, vid in videos.items()]
    for i, f in enumerate(futures.as_completed(fs)):
      # Write progress to error so that it can be seen
      sys.stderr.write( \
        "Downloaded video: {} / {} \r".format(i, len(videos)))

  print('All videos (%d) downloaded' % len(videos))


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
  parser.add_argument("-p", "--prctdwnld", type=int, metavar="PERCENTAGEDWNLD",
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
