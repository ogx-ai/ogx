import { useEffect, useRef, useState } from "react";

import { recordAudio } from "@/lib/audio-utils";

interface UseAudioRecordingOptions {
  transcribeAudio?: (blob: Blob) => Promise<string>;
  onTranscriptionComplete?: (text: string) => void;
}

export function useAudioRecording({
  transcribeAudio,
  onTranscriptionComplete,
}: UseAudioRecordingOptions) {
  const [isListening, setIsListening] = useState(false);
  const [isSpeechSupported, setIsSpeechSupported] = useState(!!transcribeAudio);
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [audioStream, setAudioStream] = useState<MediaStream | null>(null);
  const activeRecordingRef = useRef<any>(null);

  useEffect(() => {
    const checkSpeechSupport = async () => {
      const hasMediaDevices = !!(
        navigator.mediaDevices && navigator.mediaDevices.getUserMedia
      );
      setIsSpeechSupported(hasMediaDevices && !!transcribeAudio);
    };

    checkSpeechSupport();
  }, [transcribeAudio]);

  const stopRecording = async () => {
    setIsRecording(false);
    setIsTranscribing(true);
    try {
      // First stop the recording to get the final blob
      recordAudio.stop();
      // Wait for the recording promise to resolve with the final blob
      const recording = await activeRecordingRef.current;
      if (transcribeAudio) {
        const text = await transcribeAudio(recording);
        onTranscriptionComplete?.(text);
      }
    } catch (error) {
      console.error("Error transcribing audio:", error);
    } finally {
      setIsTranscribing(false);
      setIsListening(false);
      if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
        setAudioStream(null);
      }
      activeRecordingRef.current = null;
    }
  };

  const toggleListening = async () => {
    if (!isListening) {
      try {
        setIsListening(true);
        setIsRecording(true);
        // Get audio stream first
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: true,
        });
        setAudioStream(stream);

        // Start recording with the stream
        activeRecordingRef.current = recordAudio(stream);
      } catch (error) {
        console.error("Error recording audio:", error);
        setIsListening(false);
        setIsRecording(false);
        if (audioStream) {
          audioStream.getTracks().forEach(track => track.stop());
          setAudioStream(null);
        }
      }
    } else {
      await stopRecording();
    }
  };

  return {
    isListening,
    isSpeechSupported,
    isRecording,
    isTranscribing,
    audioStream,
    toggleListening,
    stopRecording,
  };
}
