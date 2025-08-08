import matplotlib.pyplot as plt
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import mediapipe as mp
import cv2
import sys
import matplotlib.colors as mcolors
import matplotlib.text as mtext

sys.path.append('../common/')
import config

#####################################
#####   PLOTTING COLORS/LABELS  #####
#####################################
colors = ['#377eb8', "green", "cyan", "red", "orchid", "darkorchid", "crimson", "lime", "fuchsia", "#ff7f00", "#f781bf", "darkcyan", "yellowgreen", "#4daf4a", "cornflowerblue",  "peru"]

blendshape_names = [
    "_neutral",
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight"
]

#####################################
####   EMBEDDING VISUALIZATION  #####
#####################################
def vis_window_sync_signal(i, main_sync_signal, bp_main_pred_interwin, plot_title = None, display = False, save_path = None):
    fig, axes = plt.subplots(2, tight_layout=True, figsize = (12, 6))
    axes[0].plot(main_sync_signal)
    if i == len(bp_main_pred_interwin) - 1:
        axes[0].vlines([bp_main_pred_interwin[i]], min(main_sync_signal), max(main_sync_signal), color = 'g', linestyle = 'dashed')
        axes[1].plot(main_sync_signal[bp_main_pred_interwin[i] - 20:])
        axes[1].vlines([20], min(main_sync_signal[bp_main_pred_interwin[i] - 20:]), max(main_sync_signal[bp_main_pred_interwin[i] - 20:]), color = 'g', linestyles = 'dashed')
    else:
        axes[0].vlines([bp_main_pred_interwin[i]], min(main_sync_signal), max(main_sync_signal), color = 'g', linestyle = 'dashed')
        axes[0].vlines([bp_main_pred_interwin[i + 1]], min(main_sync_signal), max(main_sync_signal), color = 'g', linestyle = 'dashed')
        if bp_main_pred_interwin[i] - 20 < 0:
            front_sub = 0
        else:
            front_sub = 20
        if bp_main_pred_interwin[i+1] + 20 > len(bp_main_pred_interwin):
            end_add = 0
        else:
            end_add = 20
        sig = main_sync_signal[bp_main_pred_interwin[i] - front_sub:bp_main_pred_interwin[i+1] + end_add]
        axes[1].plot(sig)
        axes[1].vlines([front_sub, len(main_sync_signal[bp_main_pred_interwin[i] - front_sub:bp_main_pred_interwin[i+1]]) - end_add], min(sig), max(sig), color = 'g', linestyles = 'dashed')
    
    if plot_title:
        title = plot_title 
    else:
        title = f"Start Pred Marker {i}"
    plt.suptitle(title, fontsize = 10)

    if display:
        plt.show()

    if save_path:
        plt.savefig(save_path)
    
    plt.close()
    
#####################################
#####   FEATURE VISUALIZATION  ######
#####################################
def annotate_frame(frame, face_bbox, detection_result, bar_graph_size = (1000, 1000), frame_size = (1000, 1000),
                    landmark_dists = None, landmark_dist_colors = None, draw_mesh = True):
    """"
    sizes are (h, w) tuples
    """
    assert bar_graph_size[0] == frame_size[0], "Bar graph and frame size must match in height"
    vis_frame = frame.copy()
    if face_bbox is not None:
        bottom = max(face_bbox[1] - config.initial_bbox_padding, 0)
        top = min(face_bbox[3]+1 + config.initial_bbox_padding, frame.shape[0])
        left = max( face_bbox[0] - config.initial_bbox_padding, 0)
        right = min(face_bbox[2] + 1 + config.initial_bbox_padding, frame.shape[1])
        vis_frame = vis_frame[bottom:top,left:right]
        if draw_mesh:
            vis_frame = draw_landmarks_on_image(vis_frame, detection_result)
            vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)

        if landmark_dists is not None:
            face_landmarks_list = detection_result.face_landmarks
            face_landmarks = face_landmarks_list[0] 
            H, W, _ = vis_frame.shape #not the same as self.input_H, self.input_W if initial face detection (and thus cropping) is being used!
            # MediaPipe by deafult returns facial landmark coordinates normalized to 0-1 based on the input frame dimensions. Here we 
            # un-normalize to get coordinates ranging from 0-W/0_H (i.e., actual pixel coordinates)
            landmark_coords = [(landmark.x * W, landmark.y * H, landmark.z) for landmark in face_landmarks] 
            for dist_num, p in enumerate(landmark_dists):
                p1, p2 = p.split("-")
                p1 = int(p1)
                p2 = int(p2)
                x1, y1, _ = landmark_coords[p1]
                x2, y2, _ = landmark_coords[p2]
                c = [chan * 255 for chan in mcolors.to_rgb(landmark_dist_colors[dist_num])[::-1]] # to RGB, then BGR as expected by OpenCV
                cv2.line(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
                
        # cv2.rectangle(vis_frame, (face_bbox[0], face_bbox[1]), (face_bbox[2], face_bbox[3]), (0, 255, 0), 4)
        # bar_graph = plot_face_blendshapes_bar_graph(detection_result.face_blendshapes[0], bar_graph_size[0], bar_graph_size[1], 
        #                                             target_blendshapes=target_blendshapes, target_blendshapes_colors=target_blendshapes_colors)
    # else:
    #     bar_graph = np.zeros((bar_graph_size[1], bar_graph_size[0], 3), dtype=np.uint8)

    vis_frame = resize_img(vis_frame, frame_size[0], frame_size[1])
    # stacked = np.hstack((vis_frame, bar_graph))
    plt.close()
    return vis_frame

def draw_landmarks_on_image(cv_image, detection_result):
  cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)  
  rgb_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_image)
  rgb_image = rgb_image.numpy_view()
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes, display_dim_w, display_dim_h, target_blendshapes = None, target_blendshapes_colors = None):
  if target_blendshapes is not None and target_blendshapes_colors is not None:
      assert len(target_blendshapes) == len(target_blendshapes_colors), f"Target blendshapes and target blendshape colors must have the same length. Currently, the lengths are {len(target_blendshapes)} and {len(target_blendshapes_colors)} respectively."
  # Extract the face blendshapes category names and scores.
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  if target_blendshapes is not None:
  # Filter the blendshapes to only include the target ones
    face_blendshapes_names = [name for i, name in enumerate(face_blendshapes_names) if i in target_blendshapes]
    face_blendshapes_scores = [score for i, score in enumerate(face_blendshapes_scores) if i in target_blendshapes]
    face_blendshapes_ranks = [rank for i, rank in enumerate(face_blendshapes_ranks) if i in target_blendshapes]
  if target_blendshapes_colors is not None:
      colors = target_blendshapes_colors
  else:
      colors = ['b' for _ in range(len(face_blendshapes_names))]  # Default to blue if no colors provided
  fig, ax = plt.subplots(figsize=(12, 12))
  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks], color = colors)
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names, fontsize=14)
  ax.invert_yaxis()

  # Label each bar with values (commenting because clutters the graph)
#   for score, patch in zip(face_blendshapes_scores, bar.patches):
#     plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top", ha="center", fontsize=12)

  ax.set_xlabel('Score', fontsize=14)
  ax.xaxis.set_tick_params(labelsize=14)
  ax.yaxis.set_tick_params(pad=15)
  ax.set_xlim(0, 1)
  ax.set_title("Face Blendshape Scores", fontsize=16)
  plt.tight_layout()
  
  figure = plt.gcf()

# set output figure size in pixels
# https://stackoverflow.com/questions/332289/how-do-i-change-the-size-of-figures-drawn-with-matplotlib/4306340#4306340
# below assumies dpi=100 
  figure.set_dpi(100)
  figure.set_size_inches(0.01*display_dim_w, 0.01*display_dim_h)
  figure.canvas.draw()
  fig_img = np.array(figure.canvas.buffer_rgba())
  fig_img = cv2.cvtColor(fig_img, cv2.COLOR_RGBA2BGR)
  plt.clf()
  return fig_img


def resize_img(img, display_dim_w, display_dim_h):
    """
    Helper function to resize inputted img to be self.display_dim x self.displa_dim pixels
    
    From https://stackoverflow.com/questions/57233910/resizing-and-padding-image-with-specific-height-and-width-in-python-opencv-gives
    """
    w, h = img.shape[1], img.shape[0]

    pad_bottom, pad_right = 0, 0
    ratio = w / h

    if h > display_dim_h or w > display_dim_w:
        # shrinking image algorithm
        interp = cv2.INTER_AREA
    else:
        # stretching image algorithm
        interp = cv2.INTER_CUBIC

    w = display_dim_w
    h = round(w / ratio)

    if h > display_dim_h:
        h = display_dim_h
        w = round(h * ratio)
    pad_top = int(abs(display_dim_h - h)/2)

    if 2*pad_top + h != display_dim_h:
        pad_bottom = pad_top + 1
    else:
        pad_bottom = pad_top
    pad_right = int(abs(display_dim_w - w)/2)

    if 2*pad_right + w != display_dim_w:
        pad_left = pad_right + 1
    else:
        pad_left = pad_right

    scaled_img = cv2.resize(img, (w, h), interpolation=interp)
    padded_img = cv2.copyMakeBorder(scaled_img,pad_top,pad_bottom,pad_left,pad_right,borderType=cv2.BORDER_CONSTANT,value=[0,0,0])
    return padded_img


class LegendTitle(object):
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(LegendTitle, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(x0, y0, r'\underline{' + orig_handle + '}', usetex=True, **self.text_props)
        handlebox.add_artist(title)
        return title
