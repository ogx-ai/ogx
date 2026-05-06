import Foundation

import LlamaStackClient

func convertToNativeSwiftType(_ value: Any) -> Any {
    switch value {
    case let number as NSNumber:
        if CFGetTypeID(number) == CFBooleanGetTypeID() {
            return number.boolValue
        }
        if floor(number.doubleValue) == number.doubleValue {
            return number.intValue
        }
        return number.doubleValue
    case let string as String:
        return string
    case let array as [Any]:
        return array.map(convertToNativeSwiftType)
    case let dict as [String: Any]:
        return dict.mapValues(convertToNativeSwiftType)
    case is NSNull:
        return NSNull()
    default:
        return value
    }
}

public class SystemDefaultGenerator {
  public init() {}

  public func gen() -> PromptTemplate {
    let templateStr = """
            Cutting Knowledge Date: December 2023
            Today Date: {{ today }}
            """

    let dateFormatter = DateFormatter()
    dateFormatter.dateFormat = "dd MMMM yyyy"

    return PromptTemplate(
      template: templateStr,
      data: ["today": dateFormatter.string(from: Date())]
    )
  }
}


public class FunctionTagCustomToolGenerator {
  public init() {}

  public func gen(customTools: [Components.Schemas.ToolDefinition]) throws -> PromptTemplate {
    // TODO: required params
    // TODO: {{#unless @last}},{{/unless}}

    let templateStr = """
            You are an expert in composing functions. You are given a question and a set of possible functions.
            Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
            If none of the function can be used, point it out. If the given question lacks the parameters required by the function,
            also point it out. You should only return the function call in tools call sections.

            If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
            You SHOULD NOT include any other text in the response.

            Here is a list of functions in JSON format that you can invoke.

            [
            {% for t in custom_tools %}
            {
                "name": "{{t.tool_name}}",
                "description": "{{t.description}}",
                "input_schema": { {{t.input_schema}} }
            }

            {{/let}}
            {% endfor -%}
            ]
            """

    let encoder = JSONEncoder()
    return PromptTemplate(
      template: templateStr,
      data: ["custom_tools": try customTools.map {
        let data = try encoder.encode($0)
        let obj = try JSONSerialization.jsonObject(with: data)
        return convertToNativeSwiftType(obj)
      }]
    )
  }
}
