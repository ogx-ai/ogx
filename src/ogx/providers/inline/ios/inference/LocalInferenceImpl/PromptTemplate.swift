import Foundation
import Stencil

public struct PromptTemplate {
    let template: String
    let data: [String: Any]

  public func render() throws -> String {
    let template = Template(templateString: self.template)
    return try template.render(self.data)
  }
}
