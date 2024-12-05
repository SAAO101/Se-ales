import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from PIL import Image
import librosa
import os
import io
import scipy.io.wavfile as wav
import tempfile
st.set_page_config(page_title="LAB3")

SAMPLING_DELTA = 0.0001
NUM_SAMPLES = 2**10

def calculate_fourier_coefficients(time_points, signal, num_harmonics, lower_limit, upper_limit):
    """
    Calculate Fourier coefficients for a periodic signal
    """
    period = upper_limit - lower_limit
    angular_freq = 2 * np.pi / period
    cosine_coeffs = np.zeros((num_harmonics, 1))
    sine_coeffs = np.zeros((num_harmonics, 1))
    num_points = len(time_points)
    
    # Calculate DC component (A0)
    dc_component = 0
    for i in range(1, num_points):
        dc_component += (1/period) * signal[i] * SAMPLING_DELTA
    
    # Calculate coefficients
    for harmonic in range(1, num_harmonics):
        for point in range(1, num_points):
            cosine_coeffs[harmonic] += ((2/period) * signal[point] * 
                                      np.cos(harmonic * angular_freq * time_points[point]) * SAMPLING_DELTA)
            sine_coeffs[harmonic] += ((2/period) * signal[point] * 
                                    np.sin(harmonic * angular_freq * time_points[point]) * SAMPLING_DELTA)
    
    return cosine_coeffs, sine_coeffs, dc_component

def compute_fourier_transform(signal):
    X_f = np.fft.fft(signal)
    X_fcorr = np.fft.fftshift(X_f)
    X_fcorr_mag = np.abs(X_fcorr)
    return X_fcorr_mag, X_fcorr

def play_audio(signal, fs):
    """Reproduce el audio usando sounddevice"""
    try:
        sd.play(signal, fs)
        sd.wait()  # Espera hasta que termine la reproducción
    except Exception as e:
        st.error(f"Error reproduciendo audio: {str(e)}")

def apply_lowpass_filter(signal, fs, cutoff_freq):
    X_f = np.fft.fft(signal)
    X_f_cent = np.fft.fftshift(X_f)
    n = len(signal)
    Delta_f = fs / n
    f = np.arange(-n/2, n/2) * Delta_f
    
    fpb = np.abs(f) <= cutoff_freq
    X_f_fil = X_f_cent * fpb
    X_f_fil_corrida = np.fft.fftshift(X_f_fil)
    x_t_filt = np.fft.ifft(X_f_fil_corrida)
    
    return np.real(x_t_filt), fpb, f

def save_audio_to_buffer(audio_signal, sample_rate):
    """Convierte la señal de audio a un buffer que puede ser reproducido por st.audio"""
    # Normalizar la señal entre -1 y 1
    normalized_signal = np.int16(audio_signal * 32767)
    buffer = io.BytesIO()
    wav.write(buffer, sample_rate, normalized_signal)
    return buffer

def perform_am_modulation(audio_file_path, carrier_freq, cutoff_freq):
    # Load audio
    x_t, fs = librosa.load(audio_file_path)
    t = np.arange(len(x_t)) / fs
    
    # Generate carrier
    carrier = np.cos(2*np.pi*carrier_freq*t)
    
    # Modulation
    y_mod = x_t * carrier
    
    # Demodulation
    x_prima = y_mod * carrier
    
    # Filtering
    x_filt, fpb, f = apply_lowpass_filter(x_prima, fs, cutoff_freq)
    
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
    
    ax1.plot(t, x_t)
    ax1.set_title("Original Signal")
    ax1.grid(True)
    
    ax2.plot(t, y_mod)
    ax2.set_title("Modulated Signal")
    ax2.grid(True)
    
    ax3.plot(t, np.real(x_filt))
    ax3.set_title("Demodulated Signal")
    ax3.grid(True)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Convertir señales a formato reproducible
    original_audio = save_audio_to_buffer(x_t, fs)
    modulated_audio = save_audio_to_buffer(y_mod, fs)
    demodulated_audio = save_audio_to_buffer(np.real(x_filt), fs)
    
    # Crear columnas para los controles de audio
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("Original Signal")
        st.audio(original_audio, format='audio/wav')
    
    with col2:
        st.write("Modulated Signal")
        st.audio(modulated_audio, format='audio/wav')
    
    with col3:
        st.write("Demodulated Signal")
        st.audio(demodulated_audio, format='audio/wav')
    
    # Mostrar información adicional
    st.write("### Signal Information")
    st.write("Original signal duration:", len(x_t)/fs, "seconds")
    st.write("Carrier frequency:", carrier_freq, "Hz")
    st.write("Cutoff frequency:", cutoff_freq, "Hz")
    
    return np.real(x_filt)

def analyze_custom_signal(time_points, num_harmonics):
    """
    Analyze the custom piecewise linear signal
    """
    T = 2  # Periodo fundamental
    
    # Generate the piecewise signal
    signal = np.piecewise(time_points, 
                         [time_points < 0, time_points >= 0],
                         [lambda t: 1 + 4*t/T, lambda t: 1 - 4*t/T])
    
    # Initialize coefficients
    a0 = 0
    an = np.zeros(num_harmonics)
    bn = np.zeros(num_harmonics)
    w0 = 2*np.pi/T
    
    # Calculate coefficients
    t_sample = np.linspace(-T/2, T/2, 1000)
    
    def integrand_a0(t):
        return np.piecewise(t, [t < 0, t >= 0],
                          [lambda t: 1 + 4*t/T, lambda t: 1 - 4*t/T])
    
    a0 = 2/T * np.trapz([integrand_a0(t) for t in t_sample], t_sample)
    
    for n in range(1, num_harmonics):
        def integrand_an(t):
            return integrand_a0(t) * np.cos(n*w0*t)
        
        def integrand_bn(t):
            return integrand_a0(t) * np.sin(n*w0*t)
        
        an[n] = 2/T * np.trapz([integrand_an(t) for t in t_sample], t_sample)
        bn[n] = 2/T * np.trapz([integrand_bn(t) for t in t_sample], t_sample)
    
    # Reconstruct signal
    reconstructed = a0/2 * np.ones_like(time_points)
    for n in range(1, num_harmonics):
        reconstructed += an[n]*np.cos(n*w0*time_points) + bn[n]*np.sin(n*w0*time_points)
    
    return signal, reconstructed, an, bn, a0/2

def analyze_sawtooth_signal(time_points, num_harmonics):
    """
    Analyze the sawtooth signal
    """
    T = 2 * np.pi
    signal = time_points % T
    mask = signal > np.pi
    signal[mask] -= T
    
    a0 = 0
    an = np.zeros(num_harmonics)
    bn = np.zeros(num_harmonics)
    w0 = 2*np.pi/T
    
    t_sample = np.linspace(-np.pi, np.pi, 1000)
    a0 = 2/T * np.trapz([t for t in t_sample], t_sample)
    
    for n in range(1, num_harmonics):
        an[n] = 2/T * np.trapz([t * np.cos(n*w0*t) for t in t_sample], t_sample)
        bn[n] = 2/T * np.trapz([t * np.sin(n*w0*t) for t in t_sample], t_sample)
    
    reconstructed = a0/2 * np.ones_like(time_points)
    for n in range(1, num_harmonics):
        reconstructed += an[n]*np.cos(n*w0*time_points) + bn[n]*np.sin(n*w0*time_points)
    
    return signal, reconstructed, an, bn, a0/2

def analyze_parabolic_signal(time_points, num_harmonics):
    """
    Analyze the parabolic signal
    """
    T = 2 * np.pi
    signal = np.zeros_like(time_points)
    
    for i, t in enumerate(time_points):
        t_norm = t % T
        if t_norm > np.pi:
            t_norm -= T
        signal[i] = t_norm**2 - np.pi**2
    
    a0 = 0
    an = np.zeros(num_harmonics)
    bn = np.zeros(num_harmonics)
    w0 = 2*np.pi/T
    
    t_sample = np.linspace(-np.pi, np.pi, 1000)
    a0 = 2/T * np.trapz([t**2 - np.pi**2 for t in t_sample], t_sample)
    
    for n in range(1, num_harmonics):
        an[n] = 2/T * np.trapz([(t**2 - np.pi**2) * np.cos(n*w0*t) for t in t_sample], t_sample)
        bn[n] = 2/T * np.trapz([(t**2 - np.pi**2) * np.sin(n*w0*t) for t in t_sample], t_sample)
    
    reconstructed = a0/2 * np.ones_like(time_points)
    for n in range(1, num_harmonics):
        reconstructed += an[n]*np.cos(n*w0*time_points) + bn[n]*np.sin(n*w0*time_points)
    
    return signal, reconstructed, an, bn, a0/2

def analyze_piecewise_new(time_points, num_harmonics):
    """
    Analyze the piecewise function
    """
    T = 2
    signal = np.piecewise(time_points, 
                         [time_points < 0, time_points >= 0],
                         [lambda t: t, lambda t: 1])
    
    a0 = 0
    an = np.zeros(num_harmonics)
    bn = np.zeros(num_harmonics)
    w0 = 2*np.pi/T
    
    t_sample = np.linspace(-1, 1, 1000)
    
    def integrand_a0(t):
        return np.piecewise(t, [t < 0, t >= 0], [lambda t: t, lambda t: 1])
    
    a0 = 2/T * np.trapz([integrand_a0(t) for t in t_sample], t_sample)
    
    for n in range(1, num_harmonics):
        an[n] = 2/T * np.trapz([integrand_a0(t) * np.cos(n*w0*t) for t in t_sample], t_sample)
        bn[n] = 2/T * np.trapz([integrand_a0(t) * np.sin(n*w0*t) for t in t_sample], t_sample)
    
    reconstructed = a0/2 * np.ones_like(time_points)
    for n in range(1, num_harmonics):
        reconstructed += an[n]*np.cos(n*w0*time_points) + bn[n]*np.sin(n*w0*time_points)
    
    return signal, reconstructed, an, bn, a0/2

def run_point5():
    signal_choice = st.selectbox(
        "Choose signal type",
        ["Custom piecewise", "Sawtooth", "Parabolic", "New piecewise"]
    )
    
    vis_choice = st.selectbox(
        "Choose visualization type",
        ["Signal comparison", "Fourier coefficients"]
    )
    
    num_harmonics = st.slider(
        "Number of harmonics",
        min_value=1,
        max_value=20,
        value=5,
        step=1
    )
    
    if signal_choice == "Custom piecewise":
        time_points = np.linspace(-1, 1, 1000)
        original, reconstructed, an, bn, a0 = analyze_custom_signal(
            time_points, num_harmonics)
    elif signal_choice == "Sawtooth":
        time_points = np.linspace(-2*np.pi, 2*np.pi, 1000)
        original, reconstructed, an, bn, a0 = analyze_sawtooth_signal(
            time_points, num_harmonics)
    elif signal_choice == "Parabolic":
        time_points = np.linspace(-3*np.pi, 3*np.pi, 1000)
        original, reconstructed, an, bn, a0 = analyze_parabolic_signal(
            time_points, num_harmonics)
    else:  # New piecewise
        time_points = np.linspace(-2, 2, 1000)
        original, reconstructed, an, bn, a0 = analyze_piecewise_new(
            time_points, num_harmonics)
    
    fig = plt.figure(figsize=(12, 6))
    
    if vis_choice == "Signal comparison":
        plt.plot(time_points, original, 'b-', label='Original', linewidth=2)
        plt.plot(time_points, reconstructed, 'r--', 
                label=f'Reconstructed ({num_harmonics} harmonics)')
        plt.title('Original vs Reconstructed Signal')
        plt.xlabel('Time (t)')
        plt.ylabel('x(t)')
        plt.grid(True)
        plt.legend()
        
        if signal_choice in ["Sawtooth", "Parabolic"]:
            plt.xticks(
                np.arange(min(time_points), max(time_points) + np.pi, np.pi),
                [f'{int(x/np.pi)}π' if x != 0 else '0' 
                 for x in np.arange(min(time_points), max(time_points) + np.pi, np.pi)]
            )
    else:
        plt.stem([0], [a0], 'b', label='DC Component')
        harmonics = np.arange(1, num_harmonics)
        plt.stem(harmonics, np.abs(an[1:]), 'r', label='|an| (Cosine)')
        plt.stem(harmonics, np.abs(bn[1:]), 'g', label='|bn| (Sine)')
        plt.title('Fourier Series Coefficients')
        plt.xlabel('n (Harmonic Number)')
        plt.ylabel('Coefficient Magnitude')
        plt.legend()
        plt.grid(True)
    
    st.pyplot(fig)

def main():
    st.title("Signal Processing Lab")
    st.subheader("Juan Polo C   Jesus Carmona   Samir Albor")
    
    choice = st.selectbox(
        "Select analysis type",
        ["None", "AM Modulation", "Fourier Series Analysis"]
    )
    
    if choice == "AM Modulation":
        try:
            # File uploader
            uploaded_file = st.file_uploader("Upload audio file", type=['wav', 'mp3'])
            
            if uploaded_file is not None:
                # Guardar temporalmente el archivo
                with open("temp_audio.wav", "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                carrier_freq = st.slider(
                    "Carrier frequency (Hz)",
                    min_value=500,
                    max_value=5000,
                    value=2000,
                    step=100
                )
                cutoff_freq = st.slider(
                    "Cutoff frequency (Hz)",
                    min_value=100,
                    max_value=1000,
                    value=700,
                    step=50
                )
                perform_am_modulation("temp_audio.wav", carrier_freq, cutoff_freq)
                
                # Eliminar el archivo temporal
                if os.path.exists("temp_audio.wav"):
                    os.remove("temp_audio.wav")
            else:
                st.info("Please upload an audio file to begin")
                
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
            if os.path.exists("temp_audio.wav"):
                os.remove("temp_audio.wav")
            
    elif choice == "Fourier Series Analysis":
        run_point5()

if __name__ == "__main__":
    main()
