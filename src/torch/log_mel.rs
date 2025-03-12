use cxx::UniquePtr;

use core::prelude::v1;
use std::path::Path;
use std::sync::Once;
use anyhow::{anyhow, Result};

#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("log_mel.h");
        
        type LogMelSpectrogram;

        fn extract(
            self: &LogMelSpectrogram,
            samples: &[f32],
        ) -> Result<()>;

        fn extract_rfft(
            self: &LogMelSpectrogram,
            samples: &[f32],
        ) -> Result<()>;

        fn log_mel_spectrogram() -> Result<UniquePtr<LogMelSpectrogram>>;
    }
}

unsafe impl Send for ffi::LogMelSpectrogram {}
unsafe impl Sync for ffi::LogMelSpectrogram {}

pub struct LogMelSpectrogram {
    ptr: UniquePtr<ffi::LogMelSpectrogram>,
}

impl LogMelSpectrogram {
    pub fn create() -> Result<Self> {
        let ptr = ffi::log_mel_spectrogram()
            .map_err(|e| anyhow!("failed to create LogMelSpectrogram: {e}"))?;
        Ok(Self { ptr })
    }

    pub fn extract(&self, samples: &[f32]) -> Result<()> {
        self.ptr.extract(samples)
            .map_err(|e| anyhow!("failed to get transcribe response: {e}"))?;
        Ok(())
    }
}