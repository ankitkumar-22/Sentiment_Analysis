import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import sounddevice as sd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from ctcdecode import CTCBeamDecoder
import os

class AdvancedSpeechModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AdvancedSpeechModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(256, hidden_dim, num_layers=4, batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, x):
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output

def load_speech_model(input_dim=80, hidden_dim=512, output_dim=29):
    model = AdvancedSpeechModel(input_dim, hidden_dim, output_dim)
    if os.path.exists("advanced_speech_model.pth"):
        model.load_state_dict(torch.load("advanced_speech_model.pth"))
    model.eval()
    return model

def record_audio(duration=5, sample_rate=16000):
    """Record audio from the microphone"""
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    return audio.flatten()

def preprocess_audio(audio, sample_rate):
    """Preprocess the audio signal"""
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=80)
    log_mel_spec = librosa.power_to_db(mel_spec)
    log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / log_mel_spec.std()
    return log_mel_spec

def decode_predictions(predictions, labels):
    decoder = CTCBeamDecoder(labels, beam_width=100, blank_id=len(labels) - 1)
    beam_results, beam_scores, timesteps, out_lens = decoder.decode(predictions)
    return ''.join([labels[p] for p in beam_results[0][0][:out_lens[0][0]]])

def speech_to_text(audio, sample_rate, model):
    """Convert speech to text using our advanced model"""
    log_mel_spec = preprocess_audio(audio, sample_rate)
    log_mel_spec_tensor = torch.FloatTensor(log_mel_spec).unsqueeze(0)
    
    with torch.no_grad():
        output = model(log_mel_spec_tensor)
        predictions = F.softmax(output, dim=-1)
    
    labels = [chr(i + 96) for i in range(1, 27)] + ['<space>', '<blank>']
    return decode_predictions(predictions, labels)

# Initialize Wav2Vec2 model
wav2vec2_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

def wav2vec2_speech_to_text(audio, sample_rate):
    """Convert speech to text using Wav2Vec2 model"""
    inputs = wav2vec2_processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        logits = wav2vec2_model(inputs.input_values, attention_mask=inputs.attention_mask).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    return wav2vec2_processor.batch_decode(predicted_ids)[0]