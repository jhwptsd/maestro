"""Convert VioPose's result.npy to Maestro's pose JSON: python src/scripts/convert_viopose.py result.npy pose.json

Use this on a result.npy you produced yourself (it is a pickle, so Maestro never loads one from an upload),
then upload pose.json to the GUI or pass it as `pose_file=` to maestro.analyze().
"""
import sys

from maestro.vision.pose_io import save_pose, viopose_result_to_pose

if __name__ == "__main__":
    src, dst = sys.argv[1:3]
    pose = viopose_result_to_pose(src)
    save_pose(pose, dst)
    print(f"{len(pose)} frames, {len(pose.joints)} joints at {pose.fps:g} fps -> {dst}")
