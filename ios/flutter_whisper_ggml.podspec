xcframework_path = File.join(__dir__, "whisper.xcframework").gsub(/ /, '\ ')

Pod::Spec.new do |s|
  s.name             = 'flutter_whisper_ggml'
  s.version          = '0.0.1'
  s.summary          = 'Flutter plugin embedding Whisper.xcframework'
  s.description      = 'Integrates a prebuilt Whisper.xcframework for on-device transcription.'
  s.homepage         = 'http://example.com'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Yawen' => 'email@example.com' }
  s.source           = { :path => '.' }

  # Code natif (bridge)
  s.source_files = 'Classes/**/*.{h,mm,cpp,swift,m}'
  s.public_header_files = 'Classes/**/*.h'

  # Flutter
  s.dependency 'Flutter'

  s.platform = :ios, '13.0'
  s.swift_version = '5.0'
  s.static_framework = false
  s.requires_arc = true

  # XCFramework embarqué
  s.vendored_frameworks = 'whisper.xcframework'
  s.preserve_paths = 'whisper.xcframework/**/*'

  s.user_target_xcconfig = {
    'STRIP_STYLE'         => 'non-global'
  }

  # ⚙️ Build settings (C++17, etc.)
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'STRIP_STYLE' => 'non-global',
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386',
    'CLANG_ENABLE_MODULES' => 'YES',
    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++17',
    'OTHER_LDFLAGS' => '$(inherited) -lc++',
    'FRAMEWORK_SEARCH_PATHS' => '$(inherited) $(PODS_CONFIGURATION_BUILD_DIR) $(PODS_XCFRAMEWORKS_BUILD_DIR)',
    'HEADER_SEARCH_PATHS' => '$(inherited) "$(PODS_CONFIGURATION_BUILD_DIR)/whisper/whisper.framework/Headers" "$(PODS_XCFRAMEWORKS_BUILD_DIR)/whisper/whisper.framework/Headers"',
    'USER_HEADER_SEARCH_PATHS' => '$(inherited) "$(PODS_CONFIGURATION_BUILD_DIR)/whisper/whisper.framework/Headers" "$(PODS_XCFRAMEWORKS_BUILD_DIR)/whisper/whisper.framework/Headers"'
  }

  # ⚡️ Forcer le linker à charger le code du framework (sinon symboles invisibles)
  s.xcconfig = {
    'OTHER_LDFLAGS[sdk=iphoneos*]' => "$(inherited) -ObjC -force_load #{xcframework_path}/ios-arm64/whisper.framework/whisper",
    'OTHER_LDFLAGS[sdk=iphonesimulator*]' => "$(inherited) -ObjC -force_load #{xcframework_path}/ios-arm64_x86_64-simulator/whisper.framework/whisper"
  }

  # Frameworks système
  s.frameworks = 'Accelerate'
end