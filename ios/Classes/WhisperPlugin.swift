import Flutter
import whisper

@objc(WhisperPlugin)
public class WhisperPlugin: NSObject, FlutterPlugin {
  public static func register(with registrar: FlutterPluginRegistrar) {
    // 👇 Empêche le linker de supprimer whisper.xcframework
    EnforceBinding.dummyMethodToEnforceBundling()

    let channel = FlutterMethodChannel(name: "whisper", binaryMessenger: registrar.messenger())
    registrar.addMethodCallDelegate(WhisperPlugin(), channel: channel)
  }

  public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    result("ready")
  }
}