"""
Deep Reinforcement Learning with Double Q-learning on Atari 2600.

COnvert mp4 to gifs.

@author: Shubham Bansal, Naman Shukla, Ziyu Zhou, Jianqiu Kong, Zhenye Na
@references:
    [1] Hado van Hasselt, Arthur Guez and David Silver.
        Deep Reinforcement Learning with Double Q-learning. arXiv:1509.06461
"""

import imageio
import os
import sys


class TargetFormat(object):
    """Target format."""

    GIF = ".gif"
    MP4 = ".mp4"
    AVI = ".avi"


def convert(inputpath, targetformat):
    """
    Convert mp4 video to gif image.

    Reference.
        http://imageio.readthedocs.io/en/latest/examples.html#convert-a-movie.

    Args:
        inputpath: .mp4 file
        targetformat: .gif file
    """
    outputpath = os.path.splitext(inputpath)[0] + targetformat
    print("converting {0} to {1}".format(inputpath, outputpath))

    reader = imageio.get_reader(inputpath)
    fps = reader.get_meta_data()['fps']

    writer = imageio.get_writer(outputpath, fps=fps)
    for i, im in enumerate(reader):
        sys.stdout.write("\rframe {0}".format(i))
        sys.stdout.flush()
        writer.append_data(im)
    print("Finalizing...")
    writer.close()
    print("Done.")


def main():
    """Convert mp4 to gifs."""

    path = "../../../Project/DDQN-pytorch-master_13/video/"

    videos_paths = ["../../../Project/DDQN-pytorch-master/video/"]
    for i in range(1, 30):
        videos_paths.append("../../../Project/DDQN-pytorch-master_{}/video/".format(i))

    # for video_path in videos_paths:
    #     all_videos = os.listdir(video_path).split(".").split("_")

    # sa
    # for video_path in videos_paths:
    convert(path + "env_90000_copy.mp4", TargetFormat.GIF)

if __name__ == '__main__':
    main()
