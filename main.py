import asyncio
import os
import sys
import pyaudio
import cv2
import traceback
from google import genai
from google.genai import types

# --- Configuration ---
# Audio Settings
INPUT_FORMAT = pyaudio.paInt16
OUTPUT_FORMAT = pyaudio.paInt16
CHANNELS = 1
INPUT_RATE = 16000
OUTPUT_RATE = 24000  # Default output sample rate for the Live API
CHUNK_SIZE = 1024

# Video Settings
FPS = 1.0  # Frames per second

# Model Definition
MODEL_ID = "gemini-2.5-flash-native-audio-preview-12-2025"  # Standard model for the Live API

# System Instruction
SYSTEM_INSTRUCTION = """You are Vigilens, an ambient, empathetic cognitive anchor and caregiver for a user with dementia. You exist in their smart glasses. You can see what they see and hear what they say. Speak in short, calming, simple sentences. If they hold an object and seem confused, gently tell them what it is and its primary use. Never rush them. If they interrupt you, stop speaking immediately and listen."""

async def capture_and_send_audio(session, audio, p_stream):
    """Captures microphone audio and streams it to the API."""
    try:
        while True:
            # Read block with exception_on_overflow=False to prevent crashes if we read too slowly
            data = await asyncio.to_thread(p_stream.read, CHUNK_SIZE, exception_on_overflow=False)
            if not data:
                await asyncio.sleep(0.01)
                continue
            
            # Send raw PCM audio to the session
            await session.send_realtime_input(
                media={"data": data, "mime_type": f"audio/pcm;rate={INPUT_RATE}"}
            )
            # Yield control back to event loop
            await asyncio.sleep(0)  
    except asyncio.CancelledError:
        print("\n[Vigilens] Audio capture task cancelled.", file=sys.stderr)
    except Exception as e:
        print(f"\n[Vigilens] Audio capture error: {e}", file=sys.stderr)
        traceback.print_exc()

async def capture_and_send_video(session):
    """Captures webcam frames and streams them to the API."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Vigilens] Error: Could not open webcam.", file=sys.stderr)
        return

    try:
        while True:
            ret, frame = await asyncio.to_thread(cap.read)
            if not ret:
                print("[Vigilens] Warning: Could not read frame from webcam.", file=sys.stderr)
                await asyncio.sleep(1.0 / FPS)
                continue

            # Encode frame to JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            success, buffer = await asyncio.to_thread(cv2.imencode, '.jpg', frame, encode_param)
            if success:
                image_bytes = buffer.tobytes()
                
                # Send frame
                await session.send_realtime_input(
                    media={"data": image_bytes, "mime_type": "image/jpeg"}
                )

            # Sleep to maintain the target framerate
            await asyncio.sleep(1.0 / FPS)
    except asyncio.CancelledError:
        print("\n[Vigilens] Video capture task cancelled.", file=sys.stderr)
    except Exception as e:
        print(f"\n[Vigilens] Video capture error: {e}", file=sys.stderr)
        traceback.print_exc()
    finally:
        cap.release()

async def receive_and_play_audio(session, audio, p_stream):
    """Receives audio response from API and plays it through speakers."""
    try:
        async for message in session.receive():
            
            if hasattr(message, 'server_content') and message.server_content:
                model_turn = message.server_content.model_turn
                if model_turn and model_turn.parts:
                    for part in model_turn.parts:
                        # Check if the part contains audio data
                        if part.inline_data and part.inline_data.data:
                            # Play the raw PCM audio chunk
                            p_stream.write(part.inline_data.data)
            
            # The model might signal that it was interrupted, you can hook into this if needed
            if hasattr(message, 'client_content_update') and message.client_content_update:
                pass
                
    except asyncio.CancelledError:
        print("\n[Vigilens] Audio playback task cancelled.", file=sys.stderr)
    except Exception as e:
        print(f"\n[Vigilens] Audio playback error: {e}", file=sys.stderr)
        traceback.print_exc()

async def main():
    # 1. Initialize API Client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set. Please set it before running.", file=sys.stderr)
        sys.exit(1)

    client = genai.Client()  # Uses GEMINI_API_KEY from environment

    # 2. Setup PyAudio
    p = pyaudio.PyAudio()
    
    try:
        input_stream = p.open(
            format=INPUT_FORMAT,
            channels=CHANNELS,
            rate=INPUT_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )
        
        output_stream = p.open(
            format=OUTPUT_FORMAT,
            channels=CHANNELS,
            rate=OUTPUT_RATE,
            output=True,
            frames_per_buffer=CHUNK_SIZE
        )
    except Exception as e:
        print(f"Error initializing audio streams: {e}", file=sys.stderr)
        p.terminate()
        sys.exit(1)

    # 3. Configure the Session
    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        system_instruction=types.Content(
            parts=[types.Part(text=SYSTEM_INSTRUCTION)]
        )
    )

    print("[Vigilens] Initializing session. Press Ctrl+C to stop...")
    
    # 4. Connect to API and run Tasks
    try:
        async with client.aio.live.connect(model=MODEL_ID, config=config) as session:
            print("[Vigilens] Connected. Awake and listening.")
            
            # Define tasks
            audio_in_task = asyncio.create_task(capture_and_send_audio(session, p, input_stream))
            video_in_task = asyncio.create_task(capture_and_send_video(session))
            audio_out_task = asyncio.create_task(receive_and_play_audio(session, p, output_stream))

            # Run all tasks concurrently
            await asyncio.gather(audio_in_task, video_in_task, audio_out_task)

    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        print("\n[Vigilens] Shutting down gracefully...", file=sys.stderr)
    except Exception as e:
        print(f"\n[Vigilens] Fatal error in main session: {e}", file=sys.stderr)
        traceback.print_exc()
    finally:
        # 5. Cleanup Resources
        print("[Vigilens] Cleaning up resources...")
        
        # Stop and close audio streams
        try:
             if input_stream.is_active():
                input_stream.stop_stream()
             input_stream.close()
        except: pass
        
        try:
            if output_stream.is_active():
                output_stream.stop_stream()
            output_stream.close()
        except: pass
        
        p.terminate()
        print("[Vigilens] Offline.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Vigilens] Exiting via Keyboard Interrupt.", file=sys.stderr)
        sys.exit(0)
