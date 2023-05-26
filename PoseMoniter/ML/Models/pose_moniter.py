import Classifier
import utils


def run(keyPoints, image_height: double, image_width: double, classification_model: str, label_file: str) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
    estimation_model: Name of the TFLite pose estimation model.
    tracker_type: Type of Tracker('keypoint' or 'bounding_box').
    classification_model: Name of the TFLite pose classification model.
      (Optional)
    label_file: Path to the label file for the pose classification model. Class
      names are listed one name per line, in the same order as in the
      classification model output. See an example in the yoga_labels.txt file.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
  """
    
  # Notify users that tracker is only enabled for MoveNet MultiPose model.
    classifier = Classifier(classification_model, label_file)
    #prob_list = classifier.classify_pose(keyPoints)

