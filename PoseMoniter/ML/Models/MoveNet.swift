// Jingbo Wang

import Accelerate
import Foundation
import TensorFlowLite
import SwiftUI

/// A wrapper to run pose estimation using MoveNet models
final class MoveNet: PoseMonitor {

  // MARK: - Private Properties
  // TensorFlow Lite `Interpreter` object for performing inference on a given model.
  private var interpreter: Interpreter

  // TensorFlow Lite `Tensor` of model input and output.
  private var inputTensor: Tensor
  private var outputTensor: Tensor

  // Model config
  private var torsoExpansionRatio = 1.9
  private var bodyExpandsionRatio = 1.2
  private let imageMean: Float = 0
  private let imageStd: Float = 1
  private let minCropKeyPointScore: Float32 = 0.2
  private var cropRegion: RectF?
  private var isProcessing = false

  // Model files
  private let movenetThunderFile = FileInfo(name: "movenet_thunder", ext: "tflite")
 
  //self mode default
  private let poseClassifierFile = FileInfo(name: "pose_classifier", ext: "tflite")
  private let labelsFile = FileInfo(name: "pose_labels", ext: "txt")
  
  // MARK: - Initialization

  /// A failable initializer for `MoveNet`. A new instance is created if the model is
  /// successfully loaded from the app's main bundle. Default `threadCount` is 4.
  init(threadCount: Int, delegate: Delegates, modelType: ModelType) throws {
    // Construct the path to the model file.
    let fileInfo: FileInfo!
    fileInfo = movenetThunderFile
    guard
      let modelPath = Bundle.main.path(
        forResource: fileInfo.name,
        ofType: fileInfo.ext)
    else {
      fatalError("Failed to load the model file with name: \(fileInfo.name).")
    }

    // Specify the options for the `Interpreter`.
    var options = Interpreter.Options()
    options.threadCount = threadCount

    // Specify the delegates for the `Interpreter`.
    var delegates: [Delegate]?
    delegates = [MetalDelegate()]

    // Create the `Interpreter`.
    interpreter = try Interpreter(modelPath: modelPath, options: options, delegates: delegates)

    // Initialize input and output `Tensor`s.
    // Allocate memory for the model's input `Tensor`s.
    try interpreter.allocateTensors()
      
    // Get allocated input and output `Tensor`s.
    inputTensor = try interpreter.input(at: 0)
    outputTensor = try interpreter.output(at: 0)
  }

  /// Runs PoseEstimation model with given image with given source area to destination area.
  /// This pose detector can process only one frame at each moment.
  ///
  /// - Parameters:
  ///   - on: Input image to run the model.
  /// - Returns: Result of the inference and the times consumed in every steps.
  func estimateSinglePose(on pixelBuffer: CVPixelBuffer) throws -> (Person, Category, Times) {
    // Check if this MoveNet instance is already processing a video frame.
    // Return an empty detection result if it's currently busy.
    guard !isProcessing else {
      throw PoseMonitorError.modelBusy
    }
    isProcessing = true
    defer {
      isProcessing = false
    }

    // Start times of each process.
    let preprocessingStartTime: Date
    let inferenceStartTime: Date
    let postprocessingStartTime: Date

    // Processing times in seconds.
    let preprocessingTime: TimeInterval
    let inferenceTime: TimeInterval
    let postprocessingTime: TimeInterval

    preprocessingStartTime = Date()
    guard let data = preprocess(pixelBuffer) else {
      os_log("Preprocessing failed.", type: .error)
      throw PoseMonitorError.preprocessingFailed
    }
    preprocessingTime = Date().timeIntervalSince(preprocessingStartTime)

    // Run inference with the TFLite model
    inferenceStartTime = Date()
    do {
      // Copy the initialized `Data` to the input `Tensor`.
      try interpreter.copy(data, toInputAt: 0)
      // Run inference by invoking the `Interpreter`.
      try interpreter.invoke()
      // Get the output `Tensor` to process the inference results.
      outputTensor = try interpreter.output(at: 0)
    } catch let error {
      os_log(
        "Failed to invoke the interpreter with error: %s", type: .error,
        error.localizedDescription)
      throw PoseMonitorError.inferenceFailed
    }
    inferenceTime = Date().timeIntervalSince(inferenceStartTime)

    postprocessingStartTime = Date()
    guard let result = postprocess(imageSize: pixelBuffer.size, modelOutput: outputTensor) else {
      os_log("Postprocessing failed.", type: .error)
      throw PoseMonitorError.postProcessingFailed
    }
    let poseList = poaseClassifier(modelFileInfo: poseClassifierFile, labelsFileInfo: labelsFile, person: result)
    postprocessingTime = Date().timeIntervalSince(postprocessingStartTime)

    let times = Times(
      preprocessing: preprocessingTime,
      inference: inferenceTime,
      postprocessing: postprocessingTime)
    return (result, poseList, times)
  }

  // MARK: - Private functions to run the model
  /// Runs PoseEstimation model with given image with given source area to destination area.
  /// This pose detector can process only one frame at each moment.
  ///
  /// - Parameters:
  ///   -modelFileInfo: own custom traning model
  ///   -labelsFileInfo: own custom pose labels file
  ///   -threadCount: default the thtread count is 4
  ///   -person: the result of the person inference
  /// - Returns: a pose name and its score
  private func poaseClassifier(modelFileInfo: FileInfo, labelsFileInfo: FileInfo, threadCount: Int = 4, person: Person)-> Category
  {
    // Construct the path to the model file.
    let fileInfo: FileInfo!
    fileInfo = modelFileInfo
    guard
      let modelPath = Bundle.main.path(
        forResource: fileInfo.name,
        ofType: fileInfo.ext)
    else {
      fatalError("Failed to load the model file with name: \(fileInfo.name).")
    }
    
    // Construct the path to the labels file.
    let labelsInfo: FileInfo!
    labelsInfo = labelsFileInfo
    guard
      let labelsPath = Bundle.main.path(
        forResource: labelsInfo.name,
        ofType: labelsInfo.ext)
    else {
      fatalError("Failed to load the labels file with name: \(labelsInfo.name).")
    }
    
    // Specify the options for the `Interpreter`.
    var options = Interpreter.Options()
    options.threadCount = threadCount
    
    let classifierInterpreter = try! Interpreter(modelPath: modelPath, options: options)
   
    // Initialize input and output `Tensor`s.
    // Allocate memory for the model's input `Tensor`s.
    try! classifierInterpreter.allocateTensors()
    
    // Get allocated input and output `self model`s.
    _ = try! classifierInterpreter.input(at: 0)
    
    let keyPoints = person.keyPoints
    var keyPointsTransed:[Float32]=[]
    for keyPoint in keyPoints
    {
      keyPointsTransed.append(Float32(keyPoint.coordinate.x))
      keyPointsTransed.append(Float32(keyPoint.coordinate.y))
      keyPointsTransed.append(keyPoint.score)
    }
    
    let data = Data(buffer: UnsafeBufferPointer(start: keyPointsTransed, count: keyPointsTransed.count))

    try! classifierInterpreter.copy(data, toInputAt: 0)
    
    try! classifierInterpreter.invoke()
    
    let outputClassifierTensor = try! classifierInterpreter.output(at: 0)
    
    let result = outputClassifierTensor.data.toArray(type: Float32.self)

    let labels = loadLabels(labelsPath: labelsPath)
    
    var labelname: String
    var labelscore: Float32
    let minimumScore: Float32 = 0.2
    if(person.score > minimumScore)
    {
      labelname = labels[argmax(result)]
      labelscore = maxscore(result)
    }
    else
    {
      labelname = "None"
      labelscore = 0.0
    }
    
    return Category(label: labelname,score: labelscore)
    
  }
  
  /// Get the index of max value in Float32 array
  ///
  /// - Parameters:
  ///   -_ array: a Float32 array
  /// - Returns: the max value's position in array
  private func argmax(_ array: [Float32]) -> Int {
    var maxValue: Float32 = -Float32.greatestFiniteMagnitude
    var maxIndex = 0
    for (index, value) in array.enumerated() {
        if value > maxValue {
            maxValue = value
            maxIndex = index
        }
    }
    return maxIndex
  }
  
  /// Get the max value in Float32 array
  ///
  /// - Parameters:
  ///   -_ array: a Float32 array
  /// - Returns: the max value in array
  private func maxscore(_ array: [Float32]) -> Float32 {
    var maxValue: Float32 = -Float32.greatestFiniteMagnitude
    for value in array
    {
      if value > maxValue
      {
          maxValue = value
      }
    }
    return maxValue
  }
  
  
  /// Load label list from file.
  /// - Parameters:
  ///   - labelsPath: Full path of label file.
  ///- Returns: An array contains the list of labels.
  private func loadLabels(labelsPath: String) -> [String]
  {
    
    let contents = try! String(contentsOfFile: labelsPath)
        return contents.components(separatedBy: .newlines)
  }
  
  /// Preprocesses given rectangle image to be `Data` of desired size by cropping and resizing it.
  ///
  /// - Parameters:
  ///   - of: Input image to crop and resize.
  /// - Returns: The cropped and resized image. `nil` if it can not be processed.
  private func preprocess(_ pixelBuffer: CVPixelBuffer) -> Data? {
    let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
    assert(
      sourcePixelFormat == kCVPixelFormatType_32BGRA
        || sourcePixelFormat == kCVPixelFormatType_32ARGB)
    // Resize `targetSquare` of input image to `modelSize`.
    let dimensions = inputTensor.shape.dimensions
    let inputWidth = CGFloat(dimensions[1])
    let inputHeight = CGFloat(dimensions[2])
    let imageWidth = pixelBuffer.size.width
    let imageHeight = pixelBuffer.size.height

    let cropRegion = self.cropRegion ??
      initialCropRegion(imageWidth: imageWidth, imageHeight: imageHeight)
    self.cropRegion = cropRegion

    let rectF = RectF(
      left: cropRegion.left * imageWidth,
      top: cropRegion.top * imageHeight,
      right: cropRegion.right * imageWidth,
      bottom: cropRegion.bottom * imageHeight)

    // Detect region
    let modelSize = CGSize(width: inputWidth, height: inputHeight)
    guard let thumbnail = pixelBuffer.cropAndResize(fromRect: rectF.rect, toSize: modelSize) else {
      return nil
    }

    // Remove the alpha component from the image buffer to get the initialized `Data`.
    guard
      let inputData = thumbnail.rgbData(
        isModelQuantized: inputTensor.dataType == .uInt8, imageMean: imageMean, imageStd: imageStd)
    else {
      os_log("Failed to convert the image buffer to RGB data.", type: .error)
      return nil
    }

    return inputData
  }

  /// Postprocesses the output `Tensor` of TFLite model to the `Person` type
  ///
  /// - Parameters:
  ///   - imageSize: Size of the input image.
  ///   - modelOutput: Output tensor from the TFLite model.
  /// - Returns: Postprocessed `Person`. `nil` if it can not be processed.
  private func postprocess(imageSize: CGSize, modelOutput: Tensor) -> Person? {
    let imageWidth = imageSize.width
    let imageHeight = imageSize.height

    let cropRegion = self.cropRegion ??
      initialCropRegion(imageWidth: imageWidth, imageHeight: imageHeight)

    let minX: CGFloat = cropRegion.left * imageWidth
    let minY: CGFloat = cropRegion.top * imageHeight

    let output = modelOutput.data.toArray(type: Float32.self)
    let dimensions = modelOutput.shape.dimensions
    let numKeyPoints = dimensions[2]
    let inputWidth = CGFloat(inputTensor.shape.dimensions[1])
    let inputHeight = CGFloat(inputTensor.shape.dimensions[2])

    let widthRatio = (cropRegion.width * imageWidth / inputWidth)
    let heightRatio = (cropRegion.height * imageHeight / inputHeight)

    // Translate the coordinates from the model output's [0..1] back to that of
    // the input image
    var positions: [CGFloat] = []
    var totalScoreSum: Float32 = 0
    var keyPoints: [KeyPoint] = []
    for idx in 0..<numKeyPoints {
      let x = ((CGFloat(output[idx * 3 + 1]) * inputWidth) * widthRatio) + minX
      let y = ((CGFloat(output[idx * 3 + 0]) * inputHeight) * heightRatio) + minY
      positions.append(x)
      positions.append(y)
      let score = output[idx * 3 + 2]
      totalScoreSum += score
      let keyPoint = KeyPoint(
        bodyPart: BodyPart.allCases[idx], coordinate: CGPoint(x: x, y: y), score: score)
      keyPoints.append(keyPoint)
    }

    // Calculate the crop region for the subsequent frame.
    self.cropRegion = nextFrameCropRegion(
      keyPoints: keyPoints, imageWidth: imageWidth, imageHeight: imageHeight)

    // Calculates total confidence score of each key position.
    let totalScore = totalScoreSum / Float32(numKeyPoints)

    // Make `Person` from `keypoints'. Each point is adjusted to the coordinate of the input image.
    return Person(keyPoints: keyPoints, score: totalScore, image_height: imageHeight, image_width: imageWidth)
  }

  // MARK: MoveNet's smart cropping logic
  /// Determines the region to crop the image for the model to run inference on.
  /// The algorithm uses the detected joints from the previous frame to estimate
  /// the square region that encloses the full body of the target person and
  /// centers at the midpoint of two hip joints. The crop size is determined by
  /// the distances between each joints and the center point.
  /// When the model is not confident with the four torso joint predictions, the
  /// function returns a default crop which is the full image padded to square.
  private func nextFrameCropRegion(keyPoints: [KeyPoint], imageWidth: CGFloat, imageHeight: CGFloat)
    -> RectF
  {
    let targetKeyPoints = keyPoints.map { keyPoint in
        KeyPoint.init(bodyPart: keyPoint.bodyPart,
                      coordinate: CGPoint(x: keyPoint.coordinate.x, y: keyPoint.coordinate.y),
                      score: keyPoint.score)
    }
    if torsoVisible(keyPoints) {
      let centerX =
        (targetKeyPoints[BodyPart.leftHip.position].coordinate.x
          + targetKeyPoints[BodyPart.rightHip.position].coordinate.x) / 2.0
      let centerY =
        (targetKeyPoints[BodyPart.leftHip.position].coordinate.y
          + targetKeyPoints[BodyPart.rightHip.position].coordinate.y) / 2.0

      let torsoAndBodyDistances =
        determineTorsoAndBodyDistances(
          keyPoints: keyPoints, targetKeyPoints: targetKeyPoints, centerX: centerX, centerY: centerY
        )

      let list = [
        torsoAndBodyDistances.maxTorsoXDistance * CGFloat(torsoExpansionRatio),
        torsoAndBodyDistances.maxTorsoYDistance * CGFloat(torsoExpansionRatio),
        torsoAndBodyDistances.maxBodyXDistance * CGFloat(bodyExpandsionRatio),
        torsoAndBodyDistances.maxBodyYDistance * CGFloat(bodyExpandsionRatio),
      ]

      var cropLengthHalf = list.max() ?? 0.0
      let tmp: [CGFloat] = [
        centerX, CGFloat(imageWidth) - centerX, centerY, CGFloat(imageHeight) - centerY,
      ]
      cropLengthHalf = min(cropLengthHalf, tmp.max() ?? 0.0)
      let cropCornerY = centerY - cropLengthHalf
      let cropCornerX = centerX - cropLengthHalf
      if cropLengthHalf > (CGFloat(max(imageWidth, imageHeight)) / 2.0) {
        return initialCropRegion(imageWidth: imageWidth, imageHeight: imageHeight)
      } else {
        let cropLength = cropLengthHalf * 2
        return RectF(
          left: max(cropCornerX, 0) / imageWidth,
          top: max(cropCornerY, 0) / imageHeight,
          right: min((cropCornerX + cropLength) / imageWidth, 1),
          bottom: min((cropCornerY + cropLength) / imageHeight, 1))
      }
    } else {
      return initialCropRegion(imageWidth: imageWidth, imageHeight: imageHeight)
    }
  }

  /// Defines the default crop region.
  /// The function provides the initial crop region (pads the full image from both
  /// sides to make it a square image) when the algorithm cannot reliably determine
  /// the crop region from the previous frame.
  private func initialCropRegion(imageWidth: CGFloat, imageHeight: CGFloat) -> RectF {
    var xMin: CGFloat
    var yMin: CGFloat
    var width: CGFloat
    var height: CGFloat
    if imageWidth > imageHeight {
      height = 1
      width = imageHeight / imageWidth
      yMin = 0
      xMin = ((imageWidth - imageHeight) / 2.0) / imageWidth
    } else {
      width = 1
      height = imageWidth / imageHeight
      xMin = 0
      yMin = ((imageHeight - imageWidth) / 2.0) / imageHeight
    }
    return RectF(left: xMin, top: yMin, right: xMin + width, bottom: yMin + height)
  }

  /// Checks whether there are enough torso keypoints.
  /// This function checks whether the model is confident at predicting one of the
  /// shoulders/hips which is required to determine a good crop region.
  private func torsoVisible(_ keyPoints: [KeyPoint]) -> Bool {
    return
      ((keyPoints[BodyPart.leftHip.position].score > minCropKeyPointScore
      || keyPoints[BodyPart.rightHip.position].score > minCropKeyPointScore))
      && ((keyPoints[BodyPart.leftShoulder.position].score > minCropKeyPointScore
        || keyPoints[BodyPart.rightShoulder.position].score > minCropKeyPointScore))
  }

  /// Calculates the maximum distance from each keypoints to the center location.
  /// The function returns the maximum distances from the two sets of keypoints:
  /// full 17 keypoints and 4 torso keypoints. The returned information will be
  /// used to determine the crop size. See determineRectF for more detail.
  private func determineTorsoAndBodyDistances(
    keyPoints: [KeyPoint], targetKeyPoints: [KeyPoint], centerX: CGFloat, centerY: CGFloat
  ) -> TorsoAndBodyDistance {
    let torsoJoints = [
      BodyPart.leftShoulder.position,
      BodyPart.rightShoulder.position,
      BodyPart.leftHip.position,
      BodyPart.rightHip.position,
    ]

    let maxTorsoYRange = torsoJoints.lazy.map { abs(centerY - targetKeyPoints[$0].coordinate.y) }
      .max() ?? 0.0
    let maxTorsoXRange = torsoJoints.lazy.map { abs(centerX - targetKeyPoints[$0].coordinate.x) }
      .max() ?? 0.0

    let confidentKeypoints = keyPoints.lazy.filter( {$0.score < self.minCropKeyPointScore} )
    let maxBodyYRange = confidentKeypoints.map({ abs(centerY - $0.coordinate.y) }).max() ?? 0.0
    let maxBodyXRange = confidentKeypoints.map({ abs(centerX - $0.coordinate.x) }).max() ?? 0.0

    return TorsoAndBodyDistance(
      maxTorsoYDistance: maxTorsoYRange,
      maxTorsoXDistance: maxTorsoXRange,
      maxBodyYDistance: maxBodyYRange,
      maxBodyXDistance: maxBodyXRange)
  }
}

/// Size of a detected person.
struct TorsoAndBodyDistance {
  var maxTorsoYDistance: CGFloat
  var maxTorsoXDistance: CGFloat
  var maxBodyYDistance: CGFloat
  var maxBodyXDistance: CGFloat
}

/// A rectangle with calculated properties for convenient access.
struct RectF {
  var left: CGFloat
  var top: CGFloat
  var right: CGFloat
  var bottom: CGFloat
  var width: CGFloat { right - left }
  var height: CGFloat { bottom - top }

  var rect: CGRect { .init(x: left, y: top, width: width, height: height) }
}
