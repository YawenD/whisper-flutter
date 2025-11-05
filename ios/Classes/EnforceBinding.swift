import Foundation

@_silgen_name("enforce_binding")
func enforce_binding()

public enum EnforceBinding {
  public static func dummyMethodToEnforceBundling() {
    enforce_binding()
  }
}

