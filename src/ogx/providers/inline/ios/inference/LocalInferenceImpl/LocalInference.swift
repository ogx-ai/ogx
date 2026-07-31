import Foundation

import LLaMARunner
import LlamaStackClient

class RunnerHolder: ObservableObject {
  var runner: Runner?
}

public class LocalInference: Inference {
  private var runnerHolder = RunnerHolder()
  private let runnerQueue: DispatchQueue

  public init (queue: DispatchQueue) {
    runnerQueue = queue
  }

  public func loadModel(modelPath: String, tokenizerPath: String, completion: @escaping (Result<Void, Error>) -> Void) {
    runnerHolder.runner = runnerHolder.runner ?? Runner(
      modelPath: modelPath,
      tokenizerPath: tokenizerPath
    )


    runnerQueue.async {
      let runner = self.runnerHolder.runner
      do {
        try runner!.load()
        completion(.success(()))
      } catch let loadError {
        print("error: " + loadError.localizedDescription)
        completion(.failure(loadError))
      }
    }
  }

  public func stop() {
    runnerHolder.runner?.stop()
  }

  public func chatCompletion(request: Components.Schemas.ChatCompletionRequest) -> AsyncStream<Components.Schemas.ChatCompletionResponseStreamChunk> {
    return AsyncStream { continuation in
      let workItem = DispatchWorkItem {
        do {
          var tokens: [String] = []

          let prompt = try encodeDialogPrompt(messages: prepareMessages(request: request))
          var stopReason: Components.Schemas.CompletionMessage.stop_reasonPayload? = nil
          var buffer = ""
          var ipython = false
          var echoDropped = false

          try self.runnerHolder.runner?.generate(prompt, sequenceLength: 4096) { token in
            buffer += token

            // HACK: Workaround until LlamaRunner exposes echo param
            if (!echoDropped) {
              if (buffer.hasPrefix(prompt)) {
                buffer = String(buffer.dropFirst(prompt.count))
                echoDropped = true
              }
              return
            }

            tokens.append(token)

            if !ipython && (buffer.starts(with: "<|python_tag|>") || buffer.starts(with: "[") ) {
              ipython = true
              continuation.yield(
                Components.Schemas.ChatCompletionResponseStreamChunk(
                  event: Components.Schemas.ChatCompletionResponseEvent(
                    event_type: .progress,
                    delta: .tool_call(Components.Schemas.ToolCallDelta(
                      _type: Components.Schemas.ToolCallDelta._typePayload.tool_call,
                      tool_call: .case1(""),
                      parse_status: Components.Schemas.ToolCallDelta.parse_statusPayload.started
                      )
                    )
                  )
                )
              )

              if (buffer.starts(with: "<|python_tag|>")) {
                buffer = String(buffer.dropFirst("<|python_tag|>".count))
              }
            }

            // TODO: Non-streaming lobprobs

            var text = ""
            if token == "<|eot_id|>" {
              stopReason = Components.Schemas.CompletionMessage.stop_reasonPayload.end_of_turn
            } else if token == "<|eom_id|>" {
              stopReason = Components.Schemas.CompletionMessage.stop_reasonPayload.end_of_message
            } else {
              text = token
            }

            var delta: Components.Schemas.ContentDelta
            if ipython {
              delta = .tool_call(Components.Schemas.ToolCallDelta(
                _type: .tool_call,
                tool_call: .case1(text),
                parse_status: .in_progress
              ))
            } else {
              delta = .text(Components.Schemas.TextDelta(
                _type: Components.Schemas.TextDelta._typePayload.text,
                text: text
                )
              )
            }

            if stopReason == nil {
              continuation.yield(
                Components.Schemas.ChatCompletionResponseStreamChunk(
                  event: Components.Schemas.ChatCompletionResponseEvent(
                    event_type: .progress,
                    delta: delta
                  )
                )
              )
            }
          }

          if stopReason == nil {
            stopReason = Components.Schemas.CompletionMessage.stop_reasonPayload.out_of_tokens
          }

          let message = decodeAssistantMessage(tokens: tokens.joined(), stopReason: stopReason!)
          // TODO: non-streaming support

          let didParseToolCalls = message.tool_calls?.count ?? 0 > 0
          if ipython && !didParseToolCalls {
            continuation.yield(
              Components.Schemas.ChatCompletionResponseStreamChunk(
                event: Components.Schemas.ChatCompletionResponseEvent(
                  event_type: .progress,
                  delta: .tool_call(Components.Schemas.ToolCallDelta(
                    _type: Components.Schemas.ToolCallDelta._typePayload.tool_call,
                    tool_call: .case1(""),
                    parse_status: Components.Schemas.ToolCallDelta.parse_statusPayload.failed
                    )
                  )
                )
                // TODO: stopReason
              )
            )
          }

          for toolCall in message.tool_calls! {
            continuation.yield(
              Components.Schemas.ChatCompletionResponseStreamChunk(
                event: Components.Schemas.ChatCompletionResponseEvent(
                  event_type: .progress,
                  delta: .tool_call(Components.Schemas.ToolCallDelta(
                    _type: Components.Schemas.ToolCallDelta._typePayload.tool_call,
                    tool_call: Components.Schemas.ToolCallDelta.tool_callPayload.ToolCall(toolCall),
                    parse_status: Components.Schemas.ToolCallDelta.parse_statusPayload.succeeded
                    )
                  )
                )
                // TODO: stopReason
              )
            )
          }

          continuation.yield(
            Components.Schemas.ChatCompletionResponseStreamChunk(
              event: Components.Schemas.ChatCompletionResponseEvent(
                event_type: .complete,
                delta: .text(Components.Schemas.TextDelta(
                  _type: Components.Schemas.TextDelta._typePayload.text,
                  text: ""
                  )
                )
              )
              // TODO: stopReason
            )
          )
        }
        catch (let error) {
          print("Inference error: " + error.localizedDescription)
        }
      }
      runnerQueue.async(execute: workItem)
    }
  }
}
