#pragma once

#include <torch/torch.h>
#include <filesystem>
#include <span>

#include "rust/cxx.h"

const int N_MELS = 128;
const int N_FFT = 400;
const int HOP_LENGTH = 160;

class LogMelSpectrogram {
    public:
        LogMelSpectrogram(
            const int nMels = N_MELS,
            const int nFFT = N_FFT,
            const int hopLength = HOP_LENGTH,
            torch::Device const& device = torch::kCUDA
        );

        torch::Tensor extract(
            const std::span<const float> samples
        ) const;

        torch::Tensor extract_rfft(
            const std::span<const float> samples
        ) const;

        void extract(
            const rust::Slice<const float> samples
        ) const {
            extract(std::span<const float>(samples.data(), samples.size()));
        };

        void extract_rfft(
            const rust::Slice<const float> samples
        ) const {
            extract_rfft(std::span<const float>(samples.data(), samples.size()));
        };

    private:
        torch::Tensor mFilters;
        torch::Tensor mWindow;
        int mNFFT;
        int mHopLength;
};

inline std::unique_ptr<LogMelSpectrogram> log_mel_spectrogram() {
    return std::make_unique<LogMelSpectrogram>();
}