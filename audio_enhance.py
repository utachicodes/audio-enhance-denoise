import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np
from scipy.signal import butter, lfilter
import os

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def normalize_audio(audio):
    max_amplitude = np.max(np.abs(audio))
    if max_amplitude > 0:
        audio = audio / max_amplitude * 0.9
    return audio

def enhance_audio(input_path, output_path, max_duration=300):
    try:
        
        audio, sr = librosa.load(input_path, sr=None, mono=True, duration=max_duration)
        
        
        if len(audio) == 0:
            raise ValueError("Audio file is empty or could not be read.")
        
        
        reduced_noise = nr.reduce_noise(y=audio, sr=sr, stationary=False, prop_decrease=0.75)
        
       
        cutoff_frequency = 8000 
        filtered_audio = lowpass_filter(reduced_noise, cutoff_frequency, sr)
        
        
        normalized_audio = normalize_audio(filtered_audio)
        
        
        sf.write(output_path, normalized_audio, sr)
        print(f"Enhanced audio saved to {output_path}")
        
        return True
    
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return False

def process_audio_file(input_file, output_dir="enhanced_audio"):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    
    file_name = os.path.splitext(os.path.basename(input_file))[0]
    output_path = os.path.join(output_dir, f"{file_name}_enhanced.wav")
    
   
    success = enhance_audio(input_file, output_path)
    
    return output_path if success else None

if __name__ == "__main__":
    
    input_audio = r"C:\Users\abdou\Downloads\vv.mp3"
    output_path = process_audio_file(input_audio)
    
    if output_path:
        print(f"Processing complete. Enhanced audio saved to {output_path}")
    else:
        print("Failed to process audio.")