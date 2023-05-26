//
//  ViewController.swift
//  PoseMoniter
//
//  Created by Jingbo Wang on 1/23/23.
//  Copyright © 2023 tensorflow and Jingbo. All rights reserved.
//

import AVFoundation
import UIKit
import os
import SwiftUI

final class ViewController: UIViewController {

  
  // MARK: Storyboards Connections
  @IBOutlet private weak var overlayView: OverlayView!
  @IBOutlet private weak var commentLabel: UILabel!
  @IBOutlet private weak var threadStepperLabel: UILabel!
  @IBOutlet private weak var threadStepper: UIStepper!
  @IBOutlet private weak var totalTimeLabel: UILabel!
  @IBOutlet private weak var scoreLabel: UILabel!
  @IBOutlet private weak var psoeNameLabel: UILabel!
  @IBOutlet private weak var delegatesSegmentedControl: UISegmentedControl!
  @IBOutlet private weak var modelSegmentedControl: UISegmentedControl!

  // MARK: Pose Moniter model configs
  private var modelType: ModelType = Constants.defaultModelType
  private var threadCount: Int = Constants.defaultThreadCount
  private var delegate: Delegates = Constants.defaultDelegate
  private let minimumScore = Constants.minimumScore
  
  private var poseQueue = Queue<String>()
  private var noneCount: Int32 = 0
  private var crosslegCount: Int32 = 0
  private var forwardheadCount: Int32 = 0
  private var standardCount: Int32 = 0
  
  private var audioPlayer = AVPlayer()
  private var warrningRingFlag = false
  
  private var ringtoneURL = Bundle.main.url(forResource: "worning", withExtension: "mp3")
  
  // MARK: Visualization
  // Relative location of `overlayView` to `previewView`.
  private var imageViewFrame: CGRect?
  // Input image overlaid with the detected keypoints.
  var overlayImage: OverlayView?

  // MARK: Controllers that manage functionality
  // Handles all data preprocessing and makes calls to run inference.
  private var poseMonitor: PoseMonitor?
  private var cameraFeedManager: CameraFeedManager!

  // Serial queue to control all tasks related to the TFLite model.
  let queue = DispatchQueue(label: "serial_queue")

  // Flag to make sure there's only one frame processed at each moment.
  var isRunning = false

  // MARK: View Handling Methods
  override func viewDidLoad() {
    super.viewDidLoad()
    updateModel()
    configCameraCapture()
  }

  override func viewWillAppear(_ animated: Bool) {
    super.viewWillAppear(animated)
    cameraFeedManager?.startRunning()
  }

  override func viewWillDisappear(_ animated: Bool) {
    super.viewWillDisappear(animated)
    cameraFeedManager?.stopRunning()
  }

  override func viewDidLayoutSubviews() {
    super.viewDidLayoutSubviews()
    imageViewFrame = overlayView.frame
  }

  private func configCameraCapture() {
    cameraFeedManager = CameraFeedManager()
    cameraFeedManager.startRunning()
    cameraFeedManager.delegate = self
  }
  
  private func updateComment(){
    let poseName = self.poseQueue.dequeue()
    switch poseName{
      case "crossleg":
        forwardheadCount = 0
        standardCount = 0
        noneCount = 0
        
        crosslegCount += 1
        
        switch crosslegCount{
          case 20...60:
            self.commentLabel.text = "Warrning: an suspected incorrect posture \nlegs crossed!"
            self.commentLabel.textColor = .orange
            self.commentLabel.font = .systemFont(ofSize: 20, weight: .bold)
            self.commentLabel.textAlignment = .center
          case let count where count > 60:
            self.commentLabel.text = "Warrning: an incorrect posture \nlegs crossed!"
            self.commentLabel.textColor = .red
            self.commentLabel.font = .systemFont(ofSize: 20, weight: .bold)
            self.commentLabel.textAlignment = .center
            warrningRingFlag = true
            if (warrningRingFlag){
              // Start the ring
              audioPlayer = AVPlayer(url: ringtoneURL!)
              audioPlayer.play()
            }
          default: break
        }

      case "forwardhead":
        crosslegCount = 0
        standardCount = 0
        noneCount = 0
      
        forwardheadCount += 1
      
        switch forwardheadCount
        {
          case 20...60:
            self.commentLabel.text = "Warrning: an suspected incorrect posture forward head!"
            self.commentLabel.textColor = .orange
            self.commentLabel.font = .systemFont(ofSize: 20, weight: .bold)
            self.commentLabel.textAlignment = .center
          case let count where count > 60:
            self.commentLabel.text = "Warrning: an incorrect posture forward head!"
            self.commentLabel.textColor = .red
            self.commentLabel.font = .systemFont(ofSize: 20, weight: .bold)
            self.commentLabel.textAlignment = .center
            warrningRingFlag = true
            if (warrningRingFlag){
              // Start the ring
              audioPlayer =  AVPlayer(url: ringtoneURL!)
              audioPlayer.play()
            }
          default: break
        }
      
      case "standard":
        crosslegCount = 0
        forwardheadCount = 0
        noneCount = 0
      
        standardCount += 1
        
        switch standardCount
        {
          case 20...60:
            self.commentLabel.text = "Standard sitting position!"
            self.commentLabel.textColor = .systemTeal
            self.commentLabel.font = .systemFont(ofSize: 20, weight: .bold)
            self.commentLabel.textAlignment = .center
          case let count where count > 60:
            self.commentLabel.text = "Good job! keep doing!"
            self.commentLabel.textColor = .green
            self.commentLabel.font = .systemFont(ofSize: 20, weight: .bold)
            self.commentLabel.textAlignment = .center
            warrningRingFlag = false
            if (!warrningRingFlag){
              // Stop the ring
              audioPlayer.pause()
              audioPlayer.seek(to: .zero)
          }
          default: break
        }

      default:
        crosslegCount = 0
        forwardheadCount = 0
        standardCount = 0
      
        noneCount += 1
        
        switch noneCount
        {
          case 20...60:
            warrningRingFlag = false
            if (!warrningRingFlag){
              // Stop the ring
              audioPlayer.pause()
              audioPlayer.seek(to: .zero)
            }
            self.commentLabel.text = "No person detected!"
            self.commentLabel.textColor = .orange
            self.commentLabel.font = .systemFont(ofSize: 20, weight: .bold)
            self.commentLabel.textAlignment = .center
          case let count where count > 60:
            self.commentLabel.text = "No more studying, right?"
            self.commentLabel.textColor = .red
            self.commentLabel.font = .systemFont(ofSize: 20, weight: .bold)
            self.commentLabel.textAlignment = .center
          default: break
        }
    }
  }
  
    
  /// Call this method when there's change in pose estimation model config, including changing model
  /// or updating runtime config.
  private func updateModel() {
    // Update the model in the same serial queue with the inference logic to avoid race condition
    queue.async {
      do {
        switch self.modelType {
        case .movenetThunder:
          self.poseMonitor = try MoveNet(
            threadCount: self.threadCount,
            delegate: self.delegate,
            modelType: self.modelType)
        }
      } catch let error {
        os_log("Error: %@", log: .default, type: .error, String(describing: error))
      }
    }
  }

  @IBAction private func threadStepperValueChanged(_ sender: UIStepper) {
    threadCount = Int(sender.value)
    threadStepperLabel.text = "\(threadCount)"
    updateModel()
  }
    
  @IBAction private func delegatesValueChanged(_ sender: UISegmentedControl) {
    delegate = Delegates.allCases[sender.selectedSegmentIndex]
    updateModel()
  }

  @IBAction private func modelTypeValueChanged(_ sender: UISegmentedControl) {
    modelType = ModelType.allCases[sender.selectedSegmentIndex]
    updateModel()
  }
}

// MARK: - CameraFeedManagerDelegate Methods
extension ViewController: CameraFeedManagerDelegate {
  func cameraFeedManager(_ cameraFeedManager: CameraFeedManager, didOutput pixelBuffer: CVPixelBuffer) {
    self.runModel(pixelBuffer)
  }

  /// Run pose estimation on the input frame from the camera.
  private func runModel(_ pixelBuffer: CVPixelBuffer)
  {
    // Guard to make sure that there's only 1 frame process at each moment.
    guard !isRunning else { return }

    // Guard to make sure that the pose estimator is already initialized.
      guard let estimator = poseMonitor else { return }

    // Run inference on a serial queue to avoid race condition.
    queue.async {
      self.isRunning = true
      defer { self.isRunning = false }
      
      var fps: Int = 0
        
      // Run pose moniter
      do{
        let (result, poseList, times) = try estimator.estimateSinglePose(
            on: pixelBuffer)
            
        // calculate fps
        fps = times.fps / 10
          
        // Return to main thread to show detection results on the app UI.
        DispatchQueue.main.async { [self] in
          self.totalTimeLabel.text = String(fps)
          
          if (poseList.score != 0.0)
          {
            self.scoreLabel.text = String(format: "%.3f", poseList.score)
          }
          else
          {
            self.scoreLabel.text = String(format: "%.1f", poseList.score)
          }
          
          switch poseList.label
          {
            case "crossleg":
              self.psoeNameLabel.text = "Cross Leg"
            case "forwardhead":
              self.psoeNameLabel.text = "Forward Head"
            case "standard":
              self.psoeNameLabel.text = "Standard"
            default:
              self.psoeNameLabel.text = poseList.label
          }
          
          // update pose queue
          self.poseQueue.enqueue(poseList.label)
          
          updateComment()
          
          // Allowed to set image and overlay
          let image = UIImage(ciImage: CIImage(cvPixelBuffer: pixelBuffer))

          // If score is too low, clear result remaining in the overlayView.
          if result.score < self.minimumScore {
            self.overlayView.image = image
            return
          }

          // Visualize the pose estimation result.
          self.overlayView.draw(at: image, person: result)
        }
        
      } catch {
        os_log("Error running pose moniter.", type: .error)
        return
      }
    }
  }
}

enum Constants {
  // Configs for the TFLite interpreter.
  static let defaultThreadCount = 4
  static let defaultDelegate: Delegates = .gpu
  static let defaultModelType: ModelType = .movenetThunder

  // Minimum score to render the result.
  static let minimumScore: Float32 = 0.2
}


struct Queue<T> {
  private var elements: [T] = []

  mutating func enqueue(_ value: T) {
    elements.append(value)
  }

  mutating func dequeue() -> T? {
    guard !elements.isEmpty else {
      return nil
    }
    return elements.removeFirst()
  }

  var head: T? {
    return elements.first
  }

  var tail: T? {
    return elements.last
  }
}
