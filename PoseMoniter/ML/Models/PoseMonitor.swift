// Jingbo Wang

import UIKit

/// Protocol to  run a pose moniter.
protocol PoseMonitor {
  func estimateSinglePose(on pixelbuffer: CVPixelBuffer) throws -> (Person, Category, Times)
}

// MARK: - Custom Errors
enum PoseMonitorError: Error {
  case modelBusy
  case preprocessingFailed
  case inferenceFailed
  case postProcessingFailed
}
