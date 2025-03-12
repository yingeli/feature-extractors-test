use std::path::Path;
use anyhow::{anyhow, Result};
use std::sync::Arc;
use tokio::fs::File;
use tokio::io::{self, AsyncReadExt, AsyncSeekExt, SeekFrom, BufReader};
use feature_extractors_test::torch::LogMelSpectrogram;
use std::time::Instant;
use futures::Future;

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

    let mut buffer: Vec<f32> = buffer.iter().take(16000*5).cloned().collect();
    buffer.extend(vec![0.0; 16000*5]);

    println!("[INFO] Read {} samples", buffer.len());

    let log_mel = LogMelSpectrogram::create()?;
    println!("[INFO] LogMelSpectrogram created");

    log_mel.extract(&buffer)?;

    let start = Instant::now();
    for _ in 0..100 {
        log_mel.extract(&buffer)?;
    }
    let duration = start.elapsed();
    println!("[INFO] Time taken for 100 executions: {:?}", duration);

    let start = Instant::now();
    for _ in 0..1000 {
        log_mel.extract(&buffer)?;
    }
    let duration = start.elapsed();
    println!("[INFO] Time taken for 1000 executions: {:?}", duration);

    Ok(())
}