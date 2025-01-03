import sounddevice as sd

# Query devices to confirm correct device ID and channels
#print(sd.query_devices())

device_id = 0  # Replace with the actual device index of the Scarlett 8i6

samplerate = 192000 #means more data to process per second
blocksize = 256 #1024, 256, 128, 64 smaller block sizes require more CPU processing power, which could lead to issues like glitches or dropouts if your system can’t handle the processing load
channels = 6  # Set to the number of input and output channels available on the Scarlett 8i6

# Callback to route input channels directly to output channels

def audio_callback(indata, outdata, frames, time, status):
    if status:
        print("Warning:", status)
    
    # Basic pass-through: route each input channel to the corresponding output channel
    outdata[:, 0] = indata[:, 0]
    outdata[:, 1] = indata[:, 0]

    outdata[:, 2] = indata[:, 1]
    outdata[:, 3] = indata[:, 1]


# Configure and start the stream
stream = sd.Stream(device=device_id, samplerate=samplerate, blocksize=blocksize, channels=channels, callback=audio_callback, latency='low')
stream.start()

try:
    print("Audio stream is active. Press Enter to stop.")
    input()

except KeyboardInterrupt:
    print("Quitting...")

finally:
    # Explicitly close the stream
    stream.stop()
    stream.close()
    print("Stream closed.")