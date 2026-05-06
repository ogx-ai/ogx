import Foundation

import LlamaStackClient

func encodeHeader(role: String) -> String {
  return "<|start_header_id|>\(role)<|end_header_id|>\n\n"
}

func encodeDialogPrompt(messages: [Components.Schemas.Message]) -> String {
  var prompt = ""

  prompt.append("<|begin_of_text|>")
  for message in messages {
    let msg = encodeMessage(message: message)
    prompt += msg
  }

  prompt.append(encodeHeader(role: "assistant"))

  return prompt
}

func getRole(message: Components.Schemas.Message) -> String {
  switch (message) {
  case .user(let m):
    return m.role.rawValue
  case .system(let m):
    return m.role.rawValue
  case .tool(let m):
    return m.role.rawValue
  case .assistant(let m):
    return m.role.rawValue
  }
}

func encodeMessage(message: Components.Schemas.Message) -> String {
  var prompt = encodeHeader(role: getRole(message: message))

  switch (message) {
  case .assistant(let m):
    if (m.tool_calls?.count ?? 0 > 0) {
      prompt += "<|python_tag|>"
    }
  default:0
    break
  }

  func _processContent(_ content: Any) -> String {
    func _process(_ c: Any) {
      if let str = c as? String {
        prompt += str
      }
    }

    if let str = content as? String {
      _process(str)
    } else if let list = content as? [Any] {
      for c in list {
        _process(c)
      }
    }

    return ""
  }

  switch (message) {
  case .user(let m):
    prompt += _processContent(m.content)
  case .system(let m):
    prompt += _processContent(m.content)
  case .tool(let m):
    prompt += _processContent(m.content)
  case .assistant(let m):
    prompt += _processContent(m.content)
  }

  var eom = false

  switch (message) {
  case .user(let m):
    switch (m.content) {
    case .case1(let c):
      prompt += _processContent(c)
    case .InterleavedContentItem(let c):
      prompt += _processContent(c)
    case .case3(let c):
      prompt += _processContent(c)
    }
  case .assistant(let m):
    // TODO: Support encoding past tool call history
    // for t in m.tool_calls {
    //  _processContent(t.)
    //}
    eom = m.stop_reason == Components.Schemas.CompletionMessage.stop_reasonPayload.end_of_message
  case .system(_):
    break
  case .tool(_):
    break
  }

  if (eom) {
    prompt += "<|eom_id|>"
  } else {
    prompt += "<|eot_id|>"
  }

  return prompt
}

func prepareMessages(request: Components.Schemas.ChatCompletionRequest) throws -> [Components.Schemas.Message] {
  var existingMessages = request.messages
  var existingSystemMessage: Components.Schemas.Message?
  // TODO: Existing system message

  var messages: [Components.Schemas.Message] = []

  let defaultGen = SystemDefaultGenerator()
  let defaultTemplate = defaultGen.gen()

  var sysContent = ""

  // TODO: Built-in tools

  sysContent += try defaultTemplate.render()

  messages.append(.system(Components.Schemas.SystemMessage(
    role: .system,
    content: .case1(sysContent)
    ))
  )

  if request.tools?.isEmpty == false {
    // TODO: Separate built-ins and custom tools (right now everything treated as custom)
    let toolGen = FunctionTagCustomToolGenerator()
    let toolTemplate = try toolGen.gen(customTools: request.tools!)
    let tools = try toolTemplate.render()
    messages.append(.user(Components.Schemas.UserMessage(
      role: .user,
      content: .case1(tools))
    ))
  }

  messages.append(contentsOf: existingMessages)

  return messages
}

struct FunctionCall {
    let name: String
    let params: [String: Any]
}

public func maybeExtractCustomToolCalls(input: String) -> [Components.Schemas.ToolCall] {
  guard input.hasPrefix("[") && input.hasSuffix("]") else {
    return []
  }

  do {
    let trimmed = input.trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
    let calls = trimmed.components(separatedBy: "),").map { $0.hasSuffix(")") ? $0 : $0 + ")" }

    var result: [Components.Schemas.ToolCall] = []

    for call in calls {
      guard let nameEndIndex = call.firstIndex(of: "("),
            let paramsStartIndex = call.firstIndex(of: "{"),
            let paramsEndIndex = call.lastIndex(of: "}") else {
        return []
      }

      let name = String(call[..<nameEndIndex]).trimmingCharacters(in: .whitespacesAndNewlines)
      let paramsString = String(call[paramsStartIndex...paramsEndIndex])

      guard let data = paramsString.data(using: .utf8),
            let params = try? JSONSerialization.jsonObject(with: data, options: []) as? [String: Any] else {
        return []
      }

      var props: [String : Components.Schemas.ToolCall.argumentsPayload.additionalPropertiesPayload] = [:]
      for (param_name, param) in params {
        switch (param) {
        case let value as String:
          props[param_name] = .case1(value)
        case let value as Int:
          props[param_name] = .case2(value)
        case let value as Double:
          props[param_name] = .case3(value)
        case let value as Bool:
          props[param_name] = .case4(value)
        default:
          return []
        }
      }

      result.append(
        Components.Schemas.ToolCall(
          call_id: UUID().uuidString,
          tool_name: .case2(name), // custom_tool
          arguments: .init(additionalProperties: props)
        )
      )
    }

    return result.isEmpty ? [] : result
  } catch {
    return []
  }
}

func decodeAssistantMessage(tokens: String, stopReason: Components.Schemas.CompletionMessage.stop_reasonPayload) -> Components.Schemas.CompletionMessage {
  var content = tokens

  let roles = ["user", "system", "assistant"]
  for role in roles {
    let headerStr = encodeHeader(role: role)
    if content.hasPrefix(headerStr) {
      content = String(content.dropFirst(encodeHeader(role: role).count))
    }
  }

  if content.hasPrefix("<|python_tag|>") {
    content = String(content.dropFirst("<|python_tag|>".count))
  }


  if content.hasSuffix("<|eot_id|>") {
    content = String(content.dropLast("<|eot_id|>".count))
  } else {
    content = String(content.dropLast("<|eom_id|>".count))
  }

  return Components.Schemas.CompletionMessage(
    role: .assistant,
    content: .case1(content),
    stop_reason: stopReason,
    tool_calls: maybeExtractCustomToolCalls(input: content)
  )
}
