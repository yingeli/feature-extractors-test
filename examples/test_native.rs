use mel_spec::prelude::*;
use std::path::Path;
use anyhow::{anyhow, Result};
use std::sync::Arc;
use tokio::fs::File;
use tokio::io::{self, AsyncReadExt, AsyncSeekExt, SeekFrom, BufReader};
use feature_extractors_test::torch::LogMelSpectrogram;

#[tokio::main]
async fn main() -> Result<()> {
    let mut audio = File::open("audio/oppo-en-30s.wav").await?;
    audio.seek(SeekFrom::Start(44)).await?;

    let mut buffer: Vec<f32> = vec![];
    loop {
        match audio.read_i16_le().await {
            Ok(sample) => buffer.push(sample as f32 / i16::MAX as f32),
            Err(e) => {
                if e.kind() != std::io::ErrorKind::UnexpectedEof {
                    return Err(anyhow!(e));
                }
                break;
            }
        }
    }

    // Take first 2000 samples and add 1000 zeros
    let mut buffer: Vec<f32> = buffer.iter().take(16000*20).cloned().collect();
    buffer.extend(vec![0.0; 16000*10]);

    let fft_size = 400;
    let sampling_rate = 16000.0;
    let n_mels = 128;
    let mut mel = MelSpectrogram::new(fft_size, sampling_rate, n_mels);

    let fft_input = Array1::from(vec![Complex::new(1.0, 0.0); fft_size]);
    // Add the FFT data to the MelSpectrogram
    let mel_spec = stage.add(fft_input);

    Ok(())
}