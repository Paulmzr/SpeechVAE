import dac
import torchaudio
from audiotools import AudioSignal
import pdb

model_path = '/data/mazhengrui/codec/descript-audio-codec/runs/vae-2/100k/vae/weights.pth'
model = dac.VAE.load(model_path)

model.to('cuda')

# Load audio signal file
signal = AudioSignal('/data/mazhengrui/dataset/LibriSpeech/test-clean/61/70968/61-70968-0000.flac')

# Encode audio signal as one long file
# (may run out of GPU memory on long files)
signal.to(model.device)

x = model.preprocess(signal.audio_data, signal.sample_rate)
posterior = model.encode(x)
z = posterior.sample()

#z, codes, latents, _, _ = model.encode(x)
# Decode audio signal
y = model.decode(z)[0].cpu()

# Write to file
torchaudio.save('output.flac', y, 16000, format='flac')